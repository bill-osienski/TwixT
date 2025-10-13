import Foundation
import Metal

/// GPU-friendly representation of TwixT game state
/// Uses flat arrays and packed data structures for Metal compute
struct GameState: Codable {
    static let boardSize: Int = 24
    static let maxPegs: Int = boardSize * boardSize
    static let maxBridges: Int = maxPegs * 8 // Each peg can have up to 8 knight-move connections

    // Board state: flat array [row * 24 + col]
    // 0 = empty, 1 = red, 2 = black
    var board: [UInt8]

    // Packed peg data: [row, col, player] for each peg
    var pegs: [PegData]
    var pegCount: Int

    // Bridge data: [from_index, to_index, player] for each bridge
    var bridges: [BridgeData]
    var bridgeCount: Int

    var currentPlayer: Player
    var moveCount: Int
    var gameOver: Bool
    var winner: Player?

    init() {
        self.board = Array(repeating: 0, count: Self.boardSize * Self.boardSize)
        self.pegs = []
        self.pegs.reserveCapacity(Self.maxPegs)
        self.pegCount = 0
        self.bridges = []
        self.bridges.reserveCapacity(Self.maxBridges)
        self.bridgeCount = 0
        self.currentPlayer = .red
        self.moveCount = 0
        self.gameOver = false
        self.winner = nil
    }

    enum Player: UInt8, Codable {
        case red = 1
        case black = 2

        var opponent: Player {
            return self == .red ? .black : .red
        }

        var name: String {
            return self == .red ? "red" : "black"
        }
    }

    struct PegData: Codable {
        let row: UInt8
        let col: UInt8
        let player: UInt8

        init(row: Int, col: Int, player: Player) {
            self.row = UInt8(row)
            self.col = UInt8(col)
            self.player = player.rawValue
        }
    }

    struct BridgeData: Codable {
        let fromRow: UInt8
        let fromCol: UInt8
        let toRow: UInt8
        let toCol: UInt8
        let player: UInt8

        init(fromRow: Int, fromCol: Int, toRow: Int, toCol: Int, player: Player) {
            self.fromRow = UInt8(fromRow)
            self.fromCol = UInt8(fromCol)
            self.toRow = UInt8(toRow)
            self.toCol = UInt8(toCol)
            self.player = player.rawValue
        }
    }

    // Fast board index calculation
    @inline(__always)
    func boardIndex(row: Int, col: Int) -> Int {
        return row * Self.boardSize + col
    }

    // Check if position is valid for current player
    func isValidPlacement(row: Int, col: Int) -> Bool {
        guard row >= 0 && row < Self.boardSize && col >= 0 && col < Self.boardSize else {
            return false
        }

        // Already occupied
        if board[boardIndex(row: row, col: col)] != 0 {
            return false
        }

        // Corners forbidden
        let isCorner = (row == 0 || row == Self.boardSize - 1) &&
                       (col == 0 || col == Self.boardSize - 1)
        if isCorner {
            return false
        }

        // Edge legality per player
        if currentPlayer == .red {
            // Red cannot place on left/right edges
            if col == 0 || col == Self.boardSize - 1 {
                return false
            }
        } else {
            // Black cannot place on top/bottom edges
            if row == 0 || row == Self.boardSize - 1 {
                return false
            }
        }

        return true
    }

    // Get all valid moves (GPU-friendly: returns indices)
    func getValidMoves() -> [Move] {
        var moves: [Move] = []
        moves.reserveCapacity(Self.boardSize * Self.boardSize - pegCount)

        for row in 0..<Self.boardSize {
            for col in 0..<Self.boardSize {
                if isValidPlacement(row: row, col: col) {
                    moves.append(Move(row: row, col: col))
                }
            }
        }

        return moves
    }

    struct Move: Codable, Hashable {
        let row: Int
        let col: Int
    }

    // Coordinate for set operations
    struct Coord: Hashable {
        let row: Int
        let col: Int
    }

    // Place a peg and create bridges
    mutating func placePeg(row: Int, col: Int) -> Bool {
        guard isValidPlacement(row: row, col: col) else {
            return false
        }

        let player = currentPlayer
        board[boardIndex(row: row, col: col)] = player.rawValue
        pegs.append(PegData(row: row, col: col, player: player))
        pegCount += 1
        moveCount += 1

        // Create bridges
        createBridges(fromRow: row, fromCol: col, player: player)

        // Check win condition
        if checkWin(player: player) {
            gameOver = true
            winner = player
            return true
        }

        // Switch player
        currentPlayer = currentPlayer.opponent
        return true
    }

    // Knight move offsets
    static let knightOffsets: [(Int, Int)] = [
        (-2, -1), (-2, 1), (-1, -2), (-1, 2),
        (1, -2), (1, 2), (2, -1), (2, 1)
    ]

    private mutating func createBridges(fromRow: Int, fromCol: Int, player: Player) {
        for (dr, dc) in Self.knightOffsets {
            let toRow = fromRow + dr
            let toCol = fromCol + dc

            guard toRow >= 0 && toRow < Self.boardSize &&
                  toCol >= 0 && toCol < Self.boardSize else {
                continue
            }

            // Check if target has same player's peg
            if board[boardIndex(row: toRow, col: toCol)] != player.rawValue {
                continue
            }

            // Check for crossing with existing bridges
            if bridgesCross(fromRow: fromRow, fromCol: fromCol, toRow: toRow, toCol: toCol) {
                continue
            }

            // Add bridge
            bridges.append(BridgeData(
                fromRow: fromRow, fromCol: fromCol,
                toRow: toRow, toCol: toCol,
                player: player
            ))
            bridgeCount += 1
        }
    }

    // Check if a potential bridge crosses existing bridges
    private func bridgesCross(fromRow: Int, fromCol: Int, toRow: Int, toCol: Int) -> Bool {
        for bridge in bridges {
            // Shared endpoint is OK
            let sharesEndpoint = (fromRow == bridge.fromRow && fromCol == bridge.fromCol) ||
                                (fromRow == bridge.toRow && fromCol == bridge.toCol) ||
                                (toRow == bridge.fromRow && toCol == bridge.fromCol) ||
                                (toRow == bridge.toRow && toCol == bridge.toCol)
            if sharesEndpoint {
                continue
            }

            // Check line segment intersection
            if lineSegmentsIntersect(
                x1: fromCol, y1: fromRow, x2: toCol, y2: toRow,
                x3: Int(bridge.fromCol), y3: Int(bridge.fromRow),
                x4: Int(bridge.toCol), y4: Int(bridge.toRow)
            ) {
                return true
            }
        }
        return false
    }

    // Robust line segment intersection test
    private func lineSegmentsIntersect(x1: Int, y1: Int, x2: Int, y2: Int,
                                      x3: Int, y3: Int, x4: Int, y4: Int) -> Bool {
        func orient(_ ax: Int, _ ay: Int, _ bx: Int, _ by: Int, _ cx: Int, _ cy: Int) -> Int {
            let v = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)
            return v > 0 ? 1 : (v < 0 ? -1 : 0)
        }

        func onSegment(_ ax: Int, _ ay: Int, _ bx: Int, _ by: Int, _ cx: Int, _ cy: Int) -> Bool {
            return min(ax, bx) <= cx && cx <= max(ax, bx) &&
                   min(ay, by) <= cy && cy <= max(ay, by)
        }

        let o1 = orient(x1, y1, x2, y2, x3, y3)
        let o2 = orient(x1, y1, x2, y2, x4, y4)
        let o3 = orient(x3, y3, x4, y4, x1, y1)
        let o4 = orient(x3, y3, x4, y4, x2, y2)

        // Proper intersection
        if o1 != o2 && o3 != o4 {
            let endpointTouch = (o1 == 0 && onSegment(x1, y1, x2, y2, x3, y3)) ||
                               (o2 == 0 && onSegment(x1, y1, x2, y2, x4, y4)) ||
                               (o3 == 0 && onSegment(x3, y3, x4, y4, x1, y1)) ||
                               (o4 == 0 && onSegment(x3, y3, x4, y4, x2, y2))
            return !endpointTouch
        }

        // Collinear cases
        if o1 == 0 && onSegment(x1, y1, x2, y2, x3, y3) {
            return !(x3 == x1 && y3 == y1) && !(x3 == x2 && y3 == y2)
        }
        if o2 == 0 && onSegment(x1, y1, x2, y2, x4, y4) {
            return !(x4 == x1 && y4 == y1) && !(x4 == x2 && y4 == y2)
        }

        return false
    }

    // Check win condition using BFS through bridges
    private func checkWin(player: Player) -> Bool {
        if player == .red {
            // Red wins: path from row 0 to row 23
            for col in 0..<Self.boardSize {
                if board[boardIndex(row: 0, col: col)] == player.rawValue {
                    let component = getConnectedComponent(startRow: 0, startCol: col, player: player)
                    for coord in component {
                        if coord.row == Self.boardSize - 1 {
                            return true
                        }
                    }
                }
            }
        } else {
            // Black wins: path from col 0 to col 23
            for row in 0..<Self.boardSize {
                if board[boardIndex(row: row, col: 0)] == player.rawValue {
                    let component = getConnectedComponent(startRow: row, startCol: 0, player: player)
                    for coord in component {
                        if coord.col == Self.boardSize - 1 {
                            return true
                        }
                    }
                }
            }
        }
        return false
    }

    // BFS to find connected component through bridges
    private func getConnectedComponent(startRow: Int, startCol: Int, player: Player) -> Set<Coord> {
        var visited = Set<Coord>()
        var queue: [Coord] = [Coord(row: startRow, col: startCol)]
        var component = Set<Coord>()

        while !queue.isEmpty {
            let coord = queue.removeFirst()

            if visited.contains(coord) {
                continue
            }

            if board[boardIndex(row: coord.row, col: coord.col)] != player.rawValue {
                continue
            }

            visited.insert(coord)
            component.insert(coord)

            // Explore through bridges
            for bridge in bridges where bridge.player == player.rawValue {
                var nextRow = -1, nextCol = -1

                if bridge.fromRow == coord.row && bridge.fromCol == coord.col {
                    nextRow = Int(bridge.toRow)
                    nextCol = Int(bridge.toCol)
                } else if bridge.toRow == coord.row && bridge.toCol == coord.col {
                    nextRow = Int(bridge.fromRow)
                    nextCol = Int(bridge.fromCol)
                } else {
                    continue
                }

                let nextCoord = Coord(row: nextRow, col: nextCol)
                if !visited.contains(nextCoord) {
                    queue.append(nextCoord)
                }
            }
        }

        return component
    }
}
