const form = document.getElementById('upload-form');
const realImg = document.getElementById('realImage');
const maskImg = document.getElementById('mask');
const overlayImg = document.getElementById('overlay');
const overlayGtImg = document.getElementById('overlayGt');
const sourceRadios = document.getElementsByName('source');
const uploadInput = document.getElementById('upload-input');
const testInput = document.getElementById('test-input');
const cardMask = document.getElementById('card-mask');
const cardOverlayGt = document.getElementById('card-overlay-gt');

// Toggle hiển thị input theo lựa chọn
sourceRadios.forEach(radio => {
  radio.addEventListener('change', () => {
    if (radio.value === 'upload' && radio.checked) {
      uploadInput.style.display = 'block';
      testInput.style.display = 'none';
    } else if (radio.value === 'test' && radio.checked) {
      uploadInput.style.display = 'none';
      testInput.style.display = 'block';
    }
  });
});

// Submit form
form.addEventListener('submit', async (e) => {
  e.preventDefault();
  const formData = new FormData();
  const selectedSource = document.querySelector('input[name="source"]:checked').value;

  if (selectedSource === 'upload') {
    const fileInput = document.querySelector('input[name="image"]');
    if (!fileInput.files.length) {
      alert('Vui lòng chọn ảnh.');
      return;
    }
    formData.append('image', fileInput.files[0]);
  } else {
    const select = document.getElementById('test-select');
    if (!select.value) {
      alert('Vui lòng chọn ảnh từ tập test.');
      return;
    }
    formData.append('test_image', select.value);
  }

  try {
    const res = await fetch('/predict', { method: 'POST', body: formData });
    if (res.ok) {
      const data = await res.json();
      console.log(data); // debug

      realImg.src = data.real;

if (selectedSource === 'upload') {
  if (data.mask && maskImg) {
    maskImg.src = data.mask;              
  }
  cardMask.style.display = 'block';     
  cardOverlayGt.style.display = 'none'; 
} else {
  if (data.overlay_gt && overlayGtImg) {
    overlayGtImg.src = data.overlay_gt;   
  }
  cardOverlayGt.style.display = 'block';
  cardMask.style.display = 'none';      
}

if (data.overlay && overlayImg) {
  overlayImg.src = data.overlay;          
}

overlayImg.src = data.overlay;          // overlay predict

    } else {
      const error = await res.json();
      alert('Error: ' + (error.error || 'Unknown error'));
    }
  } catch (err) {
    alert('Request failed: ' + err.message);
  }
});
