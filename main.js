
const NUM_SAMPLE_IMAGES = 10;
function init() {
    console.log("Num images available:",  files.length);
    let randomIndices = getRandomIndices(files, NUM_SAMPLE_IMAGES);
    console.log("Random indices", randomIndices);
    loadImage(randomIndices[0]);
}


function isFakeImage(filename) {
    return filename.toLowerCase().startsWith("tp");
}

function isRealImage(filename) {
    return filename.toLowerCase().startsWith("au");
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
    return document.querySelector("real-button");
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

