# Brain tumor detection

This project aims to predict if a patient has a brain tumor or not based on mri brain images.

## Table of Contents

* [Repository Structure](#repository-structure)
* [Development](#development)

## Repository Structure

```
.
├── app                     # Streamlit app
├── bin                     # Contains entrypoints
├── data                    # Project data
├── src                     # All the project code
└── tests                   # Unit and integration tests
```

## Development

### Installation

Install poetry and different dependencies:

```
cd brain_tumor_detection
poetry install
```

### Train model

```
poetry run ...
```

### Run app

```
cd ./app
streamlit run app.py
```

Visualize on localhost 127.0.0.1.
