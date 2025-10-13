export default class Board3DRenderer {
            constructor(container, game, gameController = null) {
                this.container = container;
                this.game = game;
                this.gameController = gameController;
                this.scene = null;
                this.camera = null;
                this.renderer = null;
                this.controls = null;
                this.raycaster = new THREE.Raycaster();
                this.mouse = new THREE.Vector2();
                this.boardGroup = new THREE.Group();
                this.pegMeshes = [];
                this.bridgeMeshes = [];
                this.holeMarkers = [];
                this.mouseDown = false;
                this.mouseDownPosition = { x: 0, y: 0 };
                this.mouseDownTime = 0;
                this.init();
            }

            init() {
                // Scene setup
                this.scene = new THREE.Scene();
                this.scene.background = new THREE.Color(0x1a1a2e);

                // Camera setup - PROPERLY CENTERED AND ZOOMED IN
                const aspect = this.container.clientWidth / this.container.clientHeight;
                this.camera = new THREE.PerspectiveCamera(45, aspect, 0.1, 1000);
                // The board mesh is centered at 11.5, the holes go from 0-23
                // Camera positioned closer to fill the screen better
                this.camera.position.set(11.5, 35, 35);
                this.camera.lookAt(11.5, 0, 11.5); // Look at the actual board center

                // Renderer setup
                this.renderer = new THREE.WebGLRenderer({
                    canvas: document.getElementById('three-canvas'),
                    antialias: true
                });
                this.renderer.setSize(this.container.clientWidth, this.container.clientHeight);
                this.renderer.shadowMap.enabled = true;
                this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;

                // Controls setup - MOUSE BUTTON SEPARATION
                this.controls = new THREE.OrbitControls(this.camera, this.renderer.domElement);
                this.controls.target.set(11.5, 0, 11.5); // Force controls to center on board
                this.controls.enableDamping = true;
                this.controls.dampingFactor = 0.1;
                this.controls.enableZoom = true;
                this.controls.enableRotate = true;
                this.controls.enablePan = true;
                this.controls.maxPolarAngle = Math.PI * 0.8;

                // Use different mouse buttons for different actions
                this.controls.mouseButtons = {
                    LEFT: null, // Disable left mouse for camera (use for piece placement)
                    MIDDLE: THREE.MOUSE.PAN, // Middle mouse for pan
                    RIGHT: THREE.MOUSE.ROTATE // Right mouse for rotation
                };
                this.controls.touches = {
                    ONE: THREE.TOUCH.ROTATE,
                    TWO: THREE.TOUCH.DOLLY_PAN
                };

                // Balanced lighting for plastic material visibility
                const ambientLight = new THREE.AmbientLight(0x404040, 0.6);
                this.scene.add(ambientLight);

                const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
                directionalLight.position.set(20, 30, 10);
                directionalLight.castShadow = true;
                directionalLight.shadow.mapSize.width = 2048;
                directionalLight.shadow.mapSize.height = 2048;
                this.scene.add(directionalLight);

                // Create board
                this.createBoard();
                this.scene.add(this.boardGroup);

                // Track mouse down/up to prevent drag-and-release from placing pieces
                let mouseDownTime = 0;
                let mouseDownPos = { x: 0, y: 0 };

                this.renderer.domElement.addEventListener('mousedown', (event) => {
                    mouseDownTime = Date.now();
                    mouseDownPos = { x: event.clientX, y: event.clientY };
                });

                // LEFT CLICK HANDLER FOR PIECE PLACEMENT ONLY
                this.renderer.domElement.addEventListener('click', (event) => {
                    // Only respond to left mouse clicks (button 0)
                    if (event.button !== 0) {
                        return; // Ignore right/middle mouse clicks
                    }

                    // Don't allow piece placement if game is over
                    if (this.game.gameOver) {
                        return;
                    }

                    const rect = this.renderer.domElement.getBoundingClientRect();
                    const mouseX = ((event.clientX - rect.left) / rect.width) * 2 - 1;
                    const mouseY = -((event.clientY - rect.top) / rect.height) * 2 + 1;

                    this.raycaster.setFromCamera({ x: mouseX, y: mouseY }, this.camera);
                    const intersects = this.raycaster.intersectObjects(this.holeMarkers);

                    if (intersects.length > 0) {
                        const { row, col } = intersects[0].object.userData;
                        const placingPlayer = this.game.currentPlayer;

                        if (this.game.placePeg(row, col)) {
                            if (this.gameController && typeof this.gameController.recordMove === 'function') {
                                this.gameController.recordMove('human', placingPlayer, row, col);
                            }
                            this.updateBoard();

                            // Update UI elements
                            const moveCount = document.getElementById('move-count');
                            if (moveCount) moveCount.textContent = `Move ${this.game.moveCount}`;

                            // Update current player display
                            const playerIndicator = document.getElementById('current-player');
                            const playerName = document.getElementById('player-name');
                            if (playerIndicator && playerName) {
                                if (this.game.currentPlayer === 'red') {
                                    playerIndicator.className = 'current-player player-red';
                                    if (this.game.isAIGame) {
                                        playerName.textContent = `Red (Human vs ${this.game.aiDifficulty.charAt(0).toUpperCase() + this.game.aiDifficulty.slice(1)} AI)`;
                                    } else {
                                        playerName.textContent = 'Red (Human vs Human)';
                                    }
                                } else {
                                    playerIndicator.className = 'current-player player-black';
                                    if (this.game.isAIGame) {
                                        playerName.textContent = `Black (${this.game.aiDifficulty.charAt(0).toUpperCase() + this.game.aiDifficulty.slice(1)} AI)`;
                                    } else {
                                        playerName.textContent = 'Black (Human vs Human)';
                                    }
                                }
                            }

                            // Enable undo button
                            const undoBtn = document.getElementById('undo');
                            if (undoBtn) {
                                undoBtn.disabled = false;
                                undoBtn.removeAttribute('disabled');
                                undoBtn.style.opacity = '1';
                                undoBtn.style.cursor = 'pointer';
                                undoBtn.style.pointerEvents = 'auto';
                            }

                            // Check for winner
                            if (this.game.gameOver && this.game.winner) {
                                const winnerName = this.game.winner === 'red' ? 'Red' : 'Black';
                                const aiStatus = this.game.isAIGame && this.game.winner === this.game.aiPlayer ? ' (AI)' : '';
                                setTimeout(() => {
                                    alert(`🎉 ${winnerName}${aiStatus} Player Wins! 🎉`);
                                }, 500);
                            }

                            // Trigger AI move if needed
                            if (this.gameController && this.gameController.onPlayerMove) {
                                this.gameController.onPlayerMove(true);
                            }
                        }
                    }
                });

                window.addEventListener('resize', this.onWindowResize.bind(this));

                // Start render loop
                this.animate();
            }

            createBoard() {
                // Table surface with wood texture (underneath)
                // Modern wood texture by ForKotLow - CC0 - https://opengameart.org/content/modern-wood-seamless-textures
                const tableGeometry = new THREE.CylinderGeometry(20, 20, 1);
                const tableMaterial = new THREE.MeshLambertMaterial({ color: 0x8b4513 });

                const woodTexture = new THREE.TextureLoader().load(
                    'assets/modernWood1.jpg',
                    (texture) => {
                        texture.wrapS = THREE.RepeatWrapping;
                        texture.wrapT = THREE.RepeatWrapping;
                        texture.repeat.set(0.8, 0.8);
                        texture.minFilter = THREE.LinearMipmapLinearFilter;
                        texture.magFilter = THREE.LinearFilter;
                        texture.anisotropy = this.renderer.capabilities.getMaxAnisotropy();
                        tableMaterial.map = texture;
                        tableMaterial.color.setHex(0xffffff);
                        tableMaterial.needsUpdate = true;
                    },
                    undefined,
                    (error) => console.warn('Failed to load wood texture:', error)
                );

                const tableMesh = new THREE.Mesh(tableGeometry, tableMaterial);
                tableMesh.position.set(11.5, -1, 11.5);
                tableMesh.receiveShadow = true;
                this.boardGroup.add(tableMesh);

                // Board base - plastic surface with realistic material properties
                const boardGeometry = new THREE.BoxGeometry(24, 0.5, 24);
                const boardMaterial = new THREE.MeshPhongMaterial({
                    color: 0xffffff, // Bright white plastic color
                    shininess: 300,  // Very high shininess for obvious plastic gloss
                    specular: 0x888888, // Strong specular highlights
                    transparent: false
                });
                const boardMesh = new THREE.Mesh(boardGeometry, boardMaterial);
                boardMesh.position.set(11.5, -0.25, 11.5);
                boardMesh.receiveShadow = true;
                boardMesh.castShadow = false; // Board doesn't cast shadows
                this.boardGroup.add(boardMesh);

                // Create holes (grid markers)
                for (let row = 0; row < this.game.boardSize; row++) {
                    for (let col = 0; col < this.game.boardSize; col++) {
                        // Visual hole marker (small, what user sees)
                        const holeGeometry = new THREE.CylinderGeometry(0.1, 0.1, 0.1);
                        const holeMaterial = new THREE.MeshLambertMaterial({ color: 0x1a1a1a });
                        const holeMesh = new THREE.Mesh(holeGeometry, holeMaterial);
                        holeMesh.position.set(col, 0, row);

                        // Invisible larger click target (easier to click)
                        const clickTargetGeometry = new THREE.CylinderGeometry(0.4, 0.4, 0.2);
                        const clickTargetMaterial = new THREE.MeshBasicMaterial({
                            color: 0xff0000,
                            transparent: true,
                            opacity: 0 // Completely invisible
                        });
                        const clickTarget = new THREE.Mesh(clickTargetGeometry, clickTargetMaterial);
                        clickTarget.position.set(col, 0.1, row);
                        clickTarget.userData = { row, col, type: 'hole' };

                        // Mark invalid positions
                        if (!this.isValidPosition(row, col)) {
                            holeMaterial.color.setHex(0x0a0a0a);
                        }

                        // Add both to the board
                        this.boardGroup.add(holeMesh);
                        this.boardGroup.add(clickTarget);

                        // Only add click target to holeMarkers for raycasting
                        this.holeMarkers.push(clickTarget);
                    }
                }

                // Edge markers
                this.createEdgeMarkers();
            }

            createEdgeMarkers() {
                // Red goal lines (top and bottom) - one dot short on each end for perfect alignment
                // Goal line between row 0 and row 1 (top), and between row 22 and row 23 (bottom)
                const topGoalGeometry = new THREE.PlaneGeometry(21, 0.2); // 21 units from position 1 to 22
                const topGoalMaterial = new THREE.MeshLambertMaterial({
                    color: 0xff4757,
                    transparent: true,
                    opacity: 0.7
                });
                const topGoal = new THREE.Mesh(topGoalGeometry, topGoalMaterial);
                topGoal.position.set(11.5, 0.01, 0.5); // Between rows 0 and 1
                topGoal.rotation.x = -Math.PI / 2;
                this.boardGroup.add(topGoal);

                const bottomGoal = new THREE.Mesh(topGoalGeometry, topGoalMaterial.clone());
                bottomGoal.position.set(11.5, 0.01, 22.5); // Between rows 22 and 23
                bottomGoal.rotation.x = -Math.PI / 2;
                this.boardGroup.add(bottomGoal);

                // Black goal lines (left and right) - one dot short on each end for perfect alignment
                // Goal line between col 0 and col 1 (left), and between col 22 and col 23 (right)
                const sideGoalGeometry = new THREE.PlaneGeometry(0.2, 21); // 21 units from position 1 to 22
                const sideGoalMaterial = new THREE.MeshLambertMaterial({
                    color: 0x000000,
                    transparent: true,
                    opacity: 0.7
                });
                const leftGoal = new THREE.Mesh(sideGoalGeometry, sideGoalMaterial);
                leftGoal.position.set(0.5, 0.01, 11.5); // Between cols 0 and 1
                leftGoal.rotation.x = -Math.PI / 2;
                this.boardGroup.add(leftGoal);

                const rightGoal = new THREE.Mesh(sideGoalGeometry, sideGoalMaterial.clone());
                rightGoal.position.set(22.5, 0.01, 11.5); // Between cols 22 and 23
                rightGoal.rotation.x = -Math.PI / 2;
                this.boardGroup.add(rightGoal);
            }

            isValidPosition(row, col) {
                if ((row === 0 || row === this.game.boardSize - 1) &&
                    (col === 0 || col === this.game.boardSize - 1)) {
                    return false;
                }
                return true;
            }

            updateBoard() {
                // Clear existing pegs and bridges
                this.pegMeshes.forEach(mesh => this.boardGroup.remove(mesh));
                this.bridgeMeshes.forEach(mesh => this.boardGroup.remove(mesh));
                this.pegMeshes = [];
                this.bridgeMeshes = [];

                // Redraw pegs
                for (const peg of this.game.pegs) {
                    this.drawPeg(peg.row, peg.col, peg.player);
                }

                // Redraw bridges
                for (const bridge of this.game.bridges) {
                    this.drawBridge(bridge.from, bridge.to, bridge.player);
                }

                // Update hole materials
                this.updateHoleHighlights();
            }

            drawPeg(row, col, player) {
                // Create a group for the complete peg
                const pegGroup = new THREE.Group();

                // Main peg material
                const pegMaterial = new THREE.MeshPhongMaterial({
                    color: player === 'red' ? 0xff4757 : 0x000000, // Red vs Black like real TwixT
                    shininess: 80
                });

                // Bottom wide section - taller
                const bottomGeometry = new THREE.CylinderGeometry(0.25, 0.35, 0.4);
                const bottomSection = new THREE.Mesh(bottomGeometry, pegMaterial);
                bottomSection.position.y = 0.2;
                bottomSection.castShadow = true;
                pegGroup.add(bottomSection);

                // Middle narrow section (the "waist" of the hourglass) - taller
                const middleGeometry = new THREE.CylinderGeometry(0.2, 0.25, 0.5);
                const middleSection = new THREE.Mesh(middleGeometry, pegMaterial);
                middleSection.position.y = 0.65;
                middleSection.castShadow = true;
                pegGroup.add(middleSection);

                // Top wide section - taller
                const topGeometry = new THREE.CylinderGeometry(0.3, 0.2, 0.4);
                const topSection = new THREE.Mesh(topGeometry, pegMaterial);
                topSection.position.y = 1.1;
                topSection.castShadow = true;
                pegGroup.add(topSection);

                // Top cap - flat wider top
                const capGeometry = new THREE.CylinderGeometry(0.32, 0.3, 0.15);
                const capMaterial = new THREE.MeshPhongMaterial({
                    color: player === 'red' ? 0xe74c3c : 0x1a1a1a, // Slightly darker shade for cap
                    shininess: 100
                });
                const cap = new THREE.Mesh(capGeometry, capMaterial);
                cap.position.y = 1.375;
                cap.castShadow = true;
                pegGroup.add(cap);

                // Base ring - wider base for stability
                const baseGeometry = new THREE.CylinderGeometry(0.38, 0.38, 0.1);
                const baseMaterial = new THREE.MeshPhongMaterial({
                    color: player === 'red' ? 0xc0392b : 0x0f0f0f, // Even darker for base
                    shininess: 60
                });
                const base = new THREE.Mesh(baseGeometry, baseMaterial);
                base.position.y = 0.05;
                base.castShadow = true;
                pegGroup.add(base);

                pegGroup.position.set(col, 0, row);

                this.pegMeshes.push(pegGroup);
                this.boardGroup.add(pegGroup);
            }

            drawBridge(from, to, player) {
                const fromPos = new THREE.Vector3(from.col, 1.2, from.row); // Connect near top of tall pegs
                const toPos = new THREE.Vector3(to.col, 1.2, to.row);
                const direction = toPos.clone().sub(fromPos);
                const distance = direction.length();

                // Create bridge group
                const bridgeGroup = new THREE.Group();

                // Main bridge body - thicker like the real bridges
                const bridgeGeometry = new THREE.CylinderGeometry(0.08, 0.08, distance - 0.4); // Slightly shorter to account for end caps
                const bridgeMaterial = new THREE.MeshPhongMaterial({
                    color: player === 'red' ? 0xff4757 : 0x000000, // Red vs Black bridges
                    shininess: 90
                });
                const bridgeMesh = new THREE.Mesh(bridgeGeometry, bridgeMaterial);
                bridgeMesh.castShadow = true;
                bridgeGroup.add(bridgeMesh);

                // End caps - rounded ends like real TwixT bridges
                const endCapGeometry = new THREE.SphereGeometry(0.08, 8, 6);
                const endCapMaterial = new THREE.MeshPhongMaterial({
                    color: player === 'red' ? 0xe74c3c : 0x1a1a1a, // Slightly darker
                    shininess: 100
                });

                // First end cap
                const endCap1 = new THREE.Mesh(endCapGeometry, endCapMaterial);
                endCap1.position.y = (distance - 0.4) / 2;
                endCap1.castShadow = true;
                bridgeGroup.add(endCap1);

                // Second end cap
                const endCap2 = new THREE.Mesh(endCapGeometry, endCapMaterial);
                endCap2.position.y = -(distance - 0.4) / 2;
                endCap2.castShadow = true;
                bridgeGroup.add(endCap2);

                // Position bridge group at midpoint
                const midpoint = fromPos.clone().add(toPos).multiplyScalar(0.5);
                bridgeGroup.position.copy(midpoint);

                // Rotate bridge to align with direction
                bridgeGroup.lookAt(toPos);
                bridgeGroup.rotateX(Math.PI / 2);

                this.bridgeMeshes.push(bridgeGroup);
                this.boardGroup.add(bridgeGroup);
            }

            updateHoleHighlights() {
                for (let i = 0; i < this.holeMarkers.length; i++) {
                    const marker = this.holeMarkers[i];
                    const { row, col } = marker.userData;

                    if (this.game.isValidPegPlacement(row, col) && !this.game.gameOver) {
                        marker.material.color.setHex(0x555555);
                    } else if (this.isValidPosition(row, col)) {
                        marker.material.color.setHex(0x1a1a1a);
                    } else {
                        marker.material.color.setHex(0x0a0a0a);
                    }
                }
            }


            onWindowResize() {
                this.camera.aspect = this.container.clientWidth / this.container.clientHeight;
                this.camera.updateProjectionMatrix();
                this.renderer.setSize(this.container.clientWidth, this.container.clientHeight);
            }

            animate() {
                requestAnimationFrame(this.animate.bind(this));
                this.controls.update();
                this.renderer.render(this.scene, this.camera);
            }
        }

        
