import GameController from './game/gameController.js';

window.addEventListener('DOMContentLoaded', () => {
    const moveCount = document.getElementById('move-count');
    if (moveCount) moveCount.textContent = 'Move 1';

    new GameController();
});
