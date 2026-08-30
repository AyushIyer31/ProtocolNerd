(function () {
    const localApiUrl = "http://localhost:8001";
    const isLocalPage = ["localhost", "127.0.0.1", ""].includes(window.location.hostname);
    const configuredApiUrl =
        window.PROTOCOLSNERD_API_URL ||
        window.PROTOCOLSNERD_BACKEND_URL ||
        (isLocalPage ? null : localStorage.getItem("PROTOCOLSNERD_API_URL")) ||
        // Single-container deploys (Cloud Run / HF Spaces / Docker) serve the
        // frontend from the backend, so default to the page's own origin.
        // Local pages keep the separate localhost backend.
        (isLocalPage ? localApiUrl : window.location.origin);

    window.env = {
        FRONTEND_FLOW: {
            SITE_NAME: "ProtocolsNerd",
            SITE_LOGO: "assets/ProtocolNerd_Logo.png",
            SITE_ICON: "CN",
            SITE_TAGLINE: "Local AI document analysis with agentic and prompt-based execution modes.",
            DISCLAIMER: "This tool performs local document analysis using Ollama. No data leaves your machine. Results should be reviewed by a qualified professional.",
            QUESTION_PLACEHOLDER: "Example: Check whether this interconnection agreement complies with the uploaded regulations.",
            STYLES: {
                BACKGROUND_COLOR: "#EFF8FF",
                FONT_FAMILY: "'Roboto', sans-serif",
                SUBMIT_BUTTON_BG: "#007bff"
            },
            API_URL: configuredApiUrl
        }
    };
})();
