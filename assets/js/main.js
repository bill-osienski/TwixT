import GameController from './game/gameController.js';

console.log('main.js loaded');

function initGame() {
    console.log('Initializing game...');
    console.log('THREE available?', typeof THREE !== 'undefined');

    const moveCount = document.getElementById('move-count');
    if (moveCount) moveCount.textContent = 'Move 1';

    try {
        console.log('Creating GameController...');
        new GameController();
        console.log('GameController created successfully');
    } catch (error) {
        console.error('Error creating GameController:', error);
    }
}

// Check if DOM is already loaded
if (document.readyState === 'loading') {
    console.log('Waiting for DOMContentLoaded...');
    document.addEventListener('DOMContentLoaded', initGame);
} else {
    console.log('DOM already loaded, initializing immediately');
    initGame();
}
