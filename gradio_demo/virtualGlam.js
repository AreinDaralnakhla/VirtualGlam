const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const context = canvas.getContext('2d');
const captureButton = document.getElementById('capture');
const tryOnButton = document.getElementById('try-on');
let selectedGarment = null;
let selectedCategory = null;

let lastClickTime = 0;

const garmentIds = ['garment_m1', 'garment_m2', 'garment_m3', 'garment_m4', 'garment_m5', 'garment_m6', 'garment_m7', 'garment_m8', 'garment_m9', 'garment_m10', 'garment_m11',
'garment_f1', 'garment_f2', 'garment_f3', 'garment_f4', 'garment_f5', 'garment_f6', 'garment_f7', 'garment_f8', 'garment_f9', 'garment_f10', 'garment_f11', 'garment_f12'];

garmentIds.forEach(id => {
    const garment = document.getElementById(id);
    garment.addEventListener('click', function() {
        removeSelections();
        this.classList.add('selected');
        selectedGarment = this;
        selectedCategory = this.getAttribute('data-category');
        console.log(selectedGarment.id, selectedCategory); 
    });
});

function removeSelections() {
    garmentIds.forEach(id => {
        const garment = document.getElementById(id);
        garment.classList.remove('selected');
    });
}

handTrack.load().then(newModel => {
    model = newModel;
    beginVideo();
}
);

function beginVideo() {
    startVideo(video).then(function (status) {
        if (status) {
            runDetection();
        } else {
            console.error('Failed to start video.');
        }
    });
}

 function startVideo(video) {
    return new Promise(function (resolve, reject) {
      if (!video) {
        resolve({ status: false, msg: "please provide a valid video element" });
      }
  
      navigator.mediaDevices
        .getUserMedia({
          audio: false,
          video: {
            facingMode: "user",
            width: { ideal: 1920 },
            height: { ideal: 1080 },
            }
        })
        .then((stream) => {
          video.srcObject = stream;
          video.onloadedmetadata = () => {
            video.play();
            video.style.transform = "rotate(90deg)";
            resolve({ status: true, msg: "webcam successfully initiated." });
          };
        })
        .catch(function (err) {
          resolve({ status: false, msg: err });
        });
    });
  }

let timeoutId = null;
let countdownElement = document.getElementById('countdown'); 

function runDetection() {
    model.detect(video).then(predictions => {
        model.renderPredictions(predictions, canvas, context, video);
        if (predictions && predictions.length > 0) {
            predictions.forEach(prediction => {
                if (prediction.label === 'point' && !timeoutId) { 
                    console.log('open hand detected - > capturing image in 5 seconds');
                    let countdown = 7;
                    countdownElement.innerText = countdown; 
                    let countdownInterval = setInterval(() => {
                        countdown--;
                        countdownElement.innerText = countdown; 
                        if (countdown === 0) {
                            clearInterval(countdownInterval); 
                        }
                    }, 1000);
                    timeoutId = window.setTimeout(() => {
                        captureButton.click();
                        timeoutId = null; 
                        countdownElement.innerText = ''; 
                    }, 5000);
                }
            });
        }
        requestAnimationFrame(runDetection);
    });
}

captureButton.addEventListener('click', () => {
    const canvas2 = document.createElement('canvas');
    const context2 = canvas2.getContext('2d');

    canvas2.width = video.videoHeight; 
    canvas2.height = video.videoWidth; 

    // Rotate context
    context2.save();
    context2.translate(canvas2.width / 2, canvas2.height / 2);
    context2.rotate(-Math.PI / 2);
    context2.drawImage(video, -video.videoWidth / 2, -video.videoHeight / 2);
    context2.restore();

    // Create a new canvas for cropping
    const cropCanvas = document.createElement('canvas');
    const cropContext = cropCanvas.getContext('2d');

    cropCanvas.width = 768;
    cropCanvas.height = 1024;

    const cropTop = 623;
    const cropBottom = 273;
    const cropHeight = canvas2.height - cropTop - cropBottom;

    const startX = (canvas2.width - cropCanvas.width) / 2; 
    const startY = cropTop;

    cropContext.drawImage(canvas2, startX, startY, cropCanvas.width, cropHeight, 0, 0, cropCanvas.width, cropHeight);

    const link = document.createElement('a');
    link.download = 'cropped_human.jpg';
    link.href = cropCanvas.toDataURL('image/jpeg');
    link.click();

    cropCanvas.toBlob(blob => {
        const file = new File([blob], 'cropped_human.jpg', { type: 'image/jpeg' });
        console.log(file);
        document.humanImage = file;
    }, 'image/jpeg');
});

// if image uploaded, we make the uploaded image as human image
document.getElementById('upload').addEventListener('change', function() {
    const file = this.files[0];
    const reader = new FileReader();
    reader.onload = function(e) {
        document.humanImage = new File([e.target.result], 'human.jpg', { type: 'image/jpeg' });
    };
    reader.readAsArrayBuffer(file);
});



tryOnButton.addEventListener('click', () => {
    const humanImage = document.humanImage;
    const selectedGarmentId = selectedGarment.id; 

    console.log('Try-on button clicked');

    console.log(humanImage);
    console.log(selectedGarmentId);
    console.log(selectedCategory);

    if (!humanImage || !selectedGarmentId || !selectedCategory) {
        alert('Please capture a photo and select a garment.');
        return;
    }

    const formData = new FormData();
    formData.append('human_img', humanImage);
    formData.append('garm_img_id', selectedGarmentId);
    formData.append('category', selectedCategory);

    fetch('http://127.0.0.1:5000/tryon', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        const outputImg = `data:image/jpeg;base64,${data.output}`;
        document.getElementById('output_result').src = outputImg;
    })
    .catch((error) => {
        console.error('Error:', error);
    });
});

// Tooltip initialization
var tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'))
var tooltipList = tooltipTriggerList.map(function (tooltipTriggerEl) {
  return new bootstrap.Tooltip(tooltipTriggerEl)
})

// Rotating Quotes
let quotes = document.querySelectorAll('.quote');
let currentQuote = 0;
setInterval(() => {
    quotes[currentQuote].classList.remove('active');
    currentQuote = (currentQuote + 1) % quotes.length;
    quotes[currentQuote].classList.add('active');
}, 5000); 

// Handle file upload
document.getElementById('upload').addEventListener('change', function() {
    const file = this.files[0];
    const reader = new FileReader();
    reader.onload = function(e) {
        document.humanImage = new File([e.target.result], 'human.jpg', { type: 'image/jpeg' });
    };
    reader.readAsArrayBuffer(file);
});
