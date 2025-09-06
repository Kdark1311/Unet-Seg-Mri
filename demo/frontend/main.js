const form = document.getElementById('upload-form');
const realImg = document.getElementById('realImage');
const maskImg = document.getElementById('mask');
const overlayImg = document.getElementById('overlay');
const sourceRadios = document.getElementsByName('source');
const uploadInput = document.getElementById('upload-input');
const testInput = document.getElementById('test-input');

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
        maskImg.src = data.mask;     // hiển thị mask dự đoán
        } else {
        maskImg.src = data.mask_gt;  // hiển thị mask gốc
}
      overlayImg.src = data.overlay;      // overlay từ mask predict
    } else {
      const error = await res.json();
      alert('Error: ' + (error.error || 'Unknown error'));
    }
  } catch (err) {
    alert('Request failed: ' + err.message);
  }
});
