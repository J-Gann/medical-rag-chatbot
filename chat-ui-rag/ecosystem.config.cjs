module.exports = {
    apps: [
      {
        name: "chat-ui-rag", // Replace "index" with your app name
        script: "build/index.js", // Path to your Node.js main file
        env: {
          PORT: 8080 // Environment variable containing the port number
        }
      }
    ]
  };