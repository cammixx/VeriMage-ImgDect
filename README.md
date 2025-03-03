# AI Image Fraud Detection System

A web-based system for detecting AI-generated and manipulated images using advanced machine learning techniques.

## Features

- Upload and analyze images for potential AI generation or manipulation
- Detailed analysis of image metadata and visual characteristics
- AI pattern detection and model identification
- User-friendly web interface with comprehensive results display
- Secure file handling and validation

## Project Structure

```
FRAUD_DETECTION/
├── ai_modules/           # AI detection implementation
│   └── detector.py      # Core detection logic
├── static/              # Static files
│   └── uploads/         # Uploaded images storage
├── templates/           # HTML templates
│   ├── index.html      # Upload page
│   └── result.html     # Results display
├── models/             # AI model storage
├── dataset/           # Training and testing datasets
├── app.py             # Flask application
└── requirements.txt   # Project dependencies
```


## Usage

1. Access the web interface through your browser
2. Upload an image for analysis
3. View detailed results including:
   - Image metadata analysis
   - Visual manipulation detection
   - AI generation patterns
   - Recommendations and reliability scores

## Development

- Frontend: Located in `templates/` directory
- Backend: Main application logic in `app.py`
- AI Module: Implementation in `ai_modules/detector.py`

## Requirements

- Python 3.8+
- See requirements.txt for complete list of dependencies

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License
