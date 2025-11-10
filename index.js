document.getElementById("analyzeBtn").addEventListener("click", async () => {
  const message = document.getElementById("message").value.trim();
  const resultBox = document.getElementById("result-box");
  const resultText = document.getElementById("result-text");
  const probText = document.getElementById("prob-text");

  if (!message) {
    alert("메시지를 입력해주세요!");
    return;
  }

  try {
    const response = await fetch("http://127.0.0.1:5000/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message }),
    });

    const data = await response.json();

    if (data.error) {
      alert(data.error);
      return;
    }

    resultBox.classList.remove("hidden");
    resultText.textContent = `예측 결과: ${data.prediction}`;
    probText.textContent = `스팸 확률: ${data.spam_probability}%`;

    if (data.prediction === "스팸") {
      resultText.style.color = "#e74c3c";
    } else {
      resultText.style.color = "#2ecc71";
    }
  } catch (error) {
    console.error(error);
    alert("서버에 연결할 수 없습니다.");
  }
});
