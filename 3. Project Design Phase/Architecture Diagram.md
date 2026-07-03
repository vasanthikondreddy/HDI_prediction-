# System Architecture

## Overview

The HDI Prediction System follows a layered architecture consisting of User Interface, Flask Application, Machine Learning Model, and Dataset.

```mermaid
flowchart LR

A[User] --> B[Flask Web Application]

B --> C[Input Validation]

C --> D[Preprocessing]

D --> E[Machine Learning Model]

E --> F[Prediction]

F --> G[Display Result]

E --> H[(Saved Model)]
```

## Components

### User Interface
Accepts socio-economic indicators from users.

### Flask Backend
Processes requests and communicates with the ML model.

### Machine Learning Model
Predicts the HDI category using trained data.

### Dataset
Provides historical HDI data used for training.