
const NUM_SAMPLE_IMAGES = 10;
currentIndex = 0;
score = 0;
function init() {
    console.log("Num images available:",  files.length);
    let randomIndices = getRandomIndices(files, NUM_SAMPLE_IMAGES);
    console.log("Random indices", randomIndices);
    loadImage(randomIndices[currentIndex]);

    getFakeButton().addEventListener("click", fakeClick);
    getRealButton().addEventListener("click", realClick);
}


function isGameOver() {
    return currentIndex  >= NUM_SAMPLE_IMAGES - 1;
}

function hideFakeButton() {
    getFakeButton().style.visibility = 'hidden';
    getFakeButton().style.display = 'none';
}

function getButtonContainer() {
    return document.querySelector("#buttons-container");
}

function gameOverState() {

    hideFakeButton();
    // SHow the final score.
    clearImage();
    if (score < 5) {
    getImageContainer().innerHTML = `
          <h4>Game over. You Got a Final score of ${score} / 10.  </h4> 
        `;
    } else if (score < 8) {
        getImageContainer().innerHTML = `
          <h4>Good Job! You Got a Final score of ${score} / 10 </h4> 
        `;
    } else if (score <= 9){
        getImageContainer().innerHTML = `
          <h4>Amazing Job! You Got a Final score of ${score} / 10 </h4> 
        `;
    } else {
        getImageContainer().innerHTML = `
          <h4>Superb Job! You Got a Final score of ${score} / 10 </h4> 
        `;
    }
    getButtonContainer().innerHTML = `
      <button id="real-button" class="mdc-button">Play Again !</button>      
    `;

    getRealButton().addEventListener('click',  () => {
      location.reload();
    });

}

function fakeClick() {
    if (isCurrentImageFake()) {
        score +=1;;
    }
    moveToNextImage();

}

function realClick() {
    if (isCurrentImageReal()) {
        score += 1;
    }
    moveToNextImage();


}

function moveToNextImage() {
    let nextImageIndex = currentIndex + 1;
    currentIndex++;
    if (isGameOver()) {
        gameOverState();
        return;
    }
    if (nextImageIndex < files.length) {
        loadImage(nextImageIndex);
    }
}

function isCurrentImageFake() {
    return isFakeImage(getImageName(currentIndex));
}

function isCurrentImageReal() {
    return isRealImage(getImageName(currentIndex));
}



function isFakeImage(filename) {
    return filename.toLowerCase().startsWith("tp");
}

function isRealImage(filename) {
    return filename.toLowerCase().startsWith("au");
}

function getImageName(imageIndex) {
    return files[imageIndex];
}


function getImagePath(imageIndex) {
    let filename = files[imageIndex];
    return IMAGE_FOLDER + "/" + filename;
}


function getCurrentImage() {
    return document.querySelector("#current-image");
}

function loadImage(imageIndex) {
    let imagePath = getImagePath(imageIndex);
    let imageContainer = getImageContainer();
    imageContainer.innerHTML = `
       <p>Test Image ${currentIndex + 1} of ${NUM_SAMPLE_IMAGES} </p>
       <a href="${imagePath}" target="_blank">
         <img id="current-image" src="${imagePath}"/> 
       </a>
    `;
}

function clearImage() {
    let currentImage = getCurrentImage();
    if (currentImage) {
        currentImage.setAttribute('src', "");
    }
}

function getImageContainer() {
    return document.querySelector("#image-container");
}

function getFakeButton() {
    return document.querySelector("#fake-button");
}

function getRealButton() {
    return document.querySelector("#real-button");
}

function getRandomIndices(inputArray, numRandomElements) {

    let results = [];

    if (inputArray.length == 0) {
        return results;
    }


    for(let i = 0; i < numRandomElements; ++i) {
        while(true) {
            let randomIndex = Math.floor(Math.random()*inputArray.length);
            if (results.includes(randomIndex)) {
                continue;
            } else {
                results.push(randomIndex);
                break;
            }
        }
    }
    return results;
}
window.addEventListener("load", init);

