import express from "express";
import axios from "axios";

const app = express();
const port = 3000;

app.use(express.json());
app.use(express.static("public"));

app.post("/api/chat", async (req, res) => {
  try {
    const { message } = req.body;

    if (!message || typeof message !== "string") {
      return res.status(400).json({
        error: "A valid message is required."
      });
    }

    const response = await axios.post("http://127.0.0.1:5000/predict", {
      message
    });

    res.json({
      answer: response.data.answer
    });
  } catch (error) {
    console.error("Error communicating with Python AI:", error.message);

    res.status(500).json({
      error: "Could not get response from AI model."
    });
  }
});

app.listen(port, () => {
  console.log(`Server is up and running on http://localhost:${port}`);
});