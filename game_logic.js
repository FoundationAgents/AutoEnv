### INIT_GAME_CODE ###
function initializeGame() {
    console.log('Initializing game...');
    
    // Reset game state
    gameState.selectedPiece = null;
    gameState.puzzlePieces = [];
    gameState.goalPlatforms = [];
    gameState.platforms = [];
    gameState.moveCount = 0;
    
    // Traverse the scene to identify and categorize objects
    scene.traverse((object) => {
        if (object.isMesh || object.isGroup) {
            // Check for puzzle pieces (movable objects)
            if (object.userData.isPuzzlePiece) {
                gameState.puzzlePieces.push(object);
                // Save original material for each mesh in the object
                object.traverse((child) => {
                    if (child.isMesh && child.material) {
                        child.userData.originalMaterial = child.material;
                    }
                });
                console.log('Found puzzle piece:', object.name);
            }
            
            // Check for goal platforms (target locations)
            if (object.userData.isGoalPlatform) {
                gameState.goalPlatforms.push(object);
                console.log('Found goal platform:', object.name);
            }
            
            // Check for regular platforms
            if (object.userData.isPlatform && !object.userData.isGoalPlatform) {
                gameState.platforms.push(object);
            }
        }
    });
    
    console.log(`Game initialized: ${gameState.puzzlePieces.length} puzzle pieces, ${gameState.goalPlatforms.length} goal platforms`);
    
    // Initial win check
    checkWinCondition();
}
### END_INIT_GAME_CODE ###

### HANDLE_KEYPRESS_CODE ###
function handleKeyPress(key) {
    key = key.toLowerCase();
    
    // ESC: Deselect current piece (FIRST priority)
    if (key === 'escape') {
        if (gameState.selectedPiece) {
            deselectPiece();
        }
        return;
    }
    
    // If no piece is selected, ignore other keys
    if (!gameState.selectedPiece) {
        if (key === 'r') {
            // R without selection: restart level
            restartLevel();
        }
        return;
    }
    
    const piece = gameState.selectedPiece;
    const gridSize = 2; // Grid alignment size
    
    // WASD/Arrow keys: Move selected piece on grid
    if (key === 'w' || key === 'arrowup') {
        piece.position.z -= gridSize;
        gameState.moveCount++;
    } else if (key === 's' || key === 'arrowdown') {
        piece.position.z += gridSize;
        gameState.moveCount++;
    } else if (key === 'a' || key === 'arrowleft') {
        piece.position.x -= gridSize;
        gameState.moveCount++;
    } else if (key === 'd' || key === 'arrowright') {
        piece.position.x += gridSize;
        gameState.moveCount++;
    }
    // Space: Snap to platform and deselect
    else if (key === ' ' || key === 'space') {
        snapToPlatform(piece);
    }
    // R: Rotate piece
    else if (key === 'r') {
        piece.rotation.y += Math.PI / 2;
        gameState.moveCount++;
    }
    
    // Update move counter display
    updateMoveCounter();
    
    // Check win condition after movement
    checkWinCondition();
}
### END_HANDLE_KEYPRESS_CODE ###

### MOUSE_CLICK_CODE ###
function onMouseClick(event) {
    // Calculate mouse position in normalized device coordinates (-1 to +1)
    const rect = renderer.domElement.getBoundingClientRect();
    mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
    mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
    
    // Update raycaster with camera and mouse position
    raycaster.setFromCamera(mouse, camera);
    
    // Check for intersections with all objects in scene
    const intersects = raycaster.intersectObjects(scene.children, true);
    
    if (intersects.length > 0) {
        // Find the first intersected object that is a puzzle piece
        for (let i = 0; i < intersects.length; i++) {
            let clickedObject = intersects[i].object;
            
            // Traverse up to find the root puzzle piece
            while (clickedObject.parent && !clickedObject.userData.isPuzzlePiece) {
                clickedObject = clickedObject.parent;
                if (clickedObject === scene) break;
            }
            
            // If we found a puzzle piece, select it
            if (clickedObject.userData.isPuzzlePiece) {
                selectPiece(clickedObject);
                return;
            }
        }
    }
    
    // If clicked on empty space, deselect current piece
    if (gameState.selectedPiece) {
        deselectPiece();
    }
}
### END_MOUSE_CLICK_CODE ###

### HELPER_FUNCTIONS ###
function selectPiece(clickedObject) {
    // If clicking the same piece, deselect it
    if (gameState.selectedPiece === clickedObject) {
        deselectPiece();
        return;
    }
    
    // Deselect previous piece if any
    if (gameState.selectedPiece) {
        deselectPiece();
    }
    
    // Select new piece
    gameState.selectedPiece = clickedObject;
    
    // Highlight the selected piece
    clickedObject.traverse((child) => {
        if (child.isMesh && child.material) {
            // Create highlight material
            const highlightMaterial = child.material.clone();
            highlightMaterial.emissive = new THREE.Color(0x00ffff);
            highlightMaterial.emissiveIntensity = 0.5;
            child.material = highlightMaterial;
        }
    });
    
    console.log('Selected piece:', clickedObject.name);
}

function deselectPiece() {
    if (!gameState.selectedPiece) return;
    
    const piece = gameState.selectedPiece;
    
    // Restore original materials
    piece.traverse((child) => {
        if (child.isMesh && child.userData.originalMaterial) {
            child.material = child.userData.originalMaterial;
        }
    });
    
    gameState.selectedPiece = null;
    console.log('Deselected piece');
}

function snapToPlatform(piece) {
    // Snap to grid alignment
    piece.position.x = Math.round(piece.position.x / 2) * 2;
    piece.position.z = Math.round(piece.position.z / 2) * 2;
    
    // Deselect the piece after snapping
    deselectPiece();
    
    console.log('Snapped piece to grid');
}

function checkWinCondition() {
    if (gameState.puzzlePieces.length === 0 || gameState.goalPlatforms.length === 0) {
        return false;
    }
    
    // Check if all puzzle pieces are on goal platforms
    let piecesOnGoal = 0;
    
    for (const piece of gameState.puzzlePieces) {
        for (const goal of gameState.goalPlatforms) {
            const distance = piece.position.distanceTo(goal.position);
            if (distance < 1.5) { // Tolerance for "on platform"
                piecesOnGoal++;
                break;
            }
        }
    }
    
    // Win if all pieces are on goal platforms
    if (piecesOnGoal === gameState.puzzlePieces.length) {
        displayWinMessage();
        return true;
    }
    
    return false;
}

function displayWinMessage() {
    console.log('🎉 Level Complete! Moves:', gameState.moveCount);
    
    // Create win message overlay
    const winDiv = document.createElement('div');
    winDiv.style.position = 'absolute';
    winDiv.style.top = '50%';
    winDiv.style.left = '50%';
    winDiv.style.transform = 'translate(-50%, -50%)';
    winDiv.style.padding = '30px';
    winDiv.style.backgroundColor = 'rgba(45, 53, 97, 0.9)';
    winDiv.style.color = '#FFD93D';
    winDiv.style.fontSize = '32px';
    winDiv.style.fontWeight = 'bold';
    winDiv.style.borderRadius = '15px';
    winDiv.style.textAlign = 'center';
    winDiv.style.zIndex = '1000';
    winDiv.innerHTML = `🎉 Level Complete! 🎉<br><span style="font-size: 20px;">Moves: ${gameState.moveCount}</span><br><span style="font-size: 16px;">Press R to restart</span>`;
    winDiv.id = 'winMessage';
    document.body.appendChild(winDiv);
}

function restartLevel() {
    // Remove win message if present
    const winMsg = document.getElementById('winMessage');
    if (winMsg) {
        winMsg.remove();
    }
    
    // Reset puzzle pieces to original positions
    gameState.puzzlePieces.forEach((piece) => {
        if (piece.userData.originalPosition) {
            piece.position.copy(piece.userData.originalPosition);
            piece.rotation.copy(piece.userData.originalRotation || new THREE.Euler());
        }
    });
    
    // Deselect any selected piece
    if (gameState.selectedPiece) {
        deselectPiece();
    }
    
    // Reset move counter
    gameState.moveCount = 0;
    updateMoveCounter();
    
    console.log('Level restarted');
}

function updateMoveCounter() {
    // Update move counter display if it exists
    const moveCounter = document.getElementById('moveCounter');
    if (moveCounter) {
        moveCounter.textContent = `Moves: ${gameState.moveCount}`;
    }
}

// Save original positions of puzzle pieces on first initialization
function saveOriginalPositions() {
    gameState.puzzlePieces.forEach((piece) => {
        if (!piece.userData.originalPosition) {
            piece.userData.originalPosition = piece.position.clone();
            piece.userData.originalRotation = piece.rotation.clone();
        }
    });
}

// Call this after initial game setup
setTimeout(() => {
    saveOriginalPositions();
}, 100);
### END_HELPER_FUNCTIONS ###
