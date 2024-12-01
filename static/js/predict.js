document
  .querySelector("form")
  .addEventListener("submit", async function (event) {
    event.preventDefault(); // Prevent page reload

    // Get form data
    const formData = new FormData(this);

    // Send data to the API
    const response = await fetch("/api/accident-predict", {
      method: "POST",
      body: formData,
    });

    // Handle response
    const resultCard = document.getElementById("result-card");
    const resultMessage = document.getElementById("result-message");

    if (response.ok) {
      const data = await response.json();
      resultMessage.textContent = `Predicted Accident Risk: ${data.accident_risk}`;
      resultCard.style.backgroundColor = "#eaf8e6"; // Green for success
      resultCard.style.color = "#2d572c";
    } else {
      resultMessage.textContent =
        "An error occurred while predicting accident risk.";
      resultCard.style.backgroundColor = "#f8d7da"; // Red for error
      resultCard.style.color = "#842029";
    }

    // Make the result card visible
    resultCard.style.display = "block";
  });
