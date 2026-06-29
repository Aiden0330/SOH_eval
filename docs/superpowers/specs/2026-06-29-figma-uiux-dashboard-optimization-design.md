# Figma UI/UX Dashboard Optimization Design

## Context

The Figma review produced two directions for the SOH evaluation frontend:

- Mockup A: a training workflow that guides users through resource upload, experiment configuration, execution, and model artifact publication.
- Mockup B: a data workbench that emphasizes model and dataset comparison, compatibility checks, transfer metrics, and inference speed.

The approved product direction is to use Mockup B as the overall information architecture while using Mockup A inside the training evaluation tab.

## Goals

- Make the three tabs visually and functionally distinct.
- Keep the interface Chinese-first.
- Preserve the existing static artifact loading and backend API contracts.
- Avoid changing backend behavior in this UI pass.

## Page Roles

### Training Evaluation

The training tab is an experiment orchestration surface. It should focus on:

- Resource upload for datasets, algorithm source plugins, and feature source plugins.
- Training matrix setup: datasets, algorithms, featuresets, protocols, and seeds.
- Estimated combination count and readiness.
- Job progress and artifact publication.
- Clear indication that training can produce Pickle models for the model library.

### Inference Evaluation

The inference tab is a transfer/generalization analysis surface. It should focus on:

- Pickle model selection and model upload.
- Target dataset selection.
- Compatibility precheck messaging.
- Model x dataset inference results.
- MAE/RMSE, predict time, ms per 1000 samples, and throughput.
- Automatic interpretation of best transfer, fastest model, and failed rows.

### Model Library

The model library is an asset governance surface. It should focus on:

- Model source: training-generated or uploaded Pickle.
- Estimator, featureset, trained datasets, and feature column count.
- Recent inference performance when available.
- A clear next action that sends users to inference evaluation rather than duplicating the inference selector.

## Layout Decisions

- Keep the top-level KPI band and global scope context for overall situational awareness.
- Convert the active tab panel into the first meaningful working area below the tab switcher.
- Training uses a three-column layout: workflow steps, configuration matrix, artifact status.
- Inference uses a three-column layout: inputs and compatibility, results matrix, interpretation.
- Model library uses an asset table with summary cards and a short governance-oriented empty state.

## Visual Decisions

- Use restrained workbench styling rather than marketing layout.
- Use compact cards with 8px radii for repeated items and tool panels.
- Use tab-specific accent colors:
  - Training: blue.
  - Inference: green.
  - Model library: amber.
- Avoid nested decorative cards where a dense operational layout is clearer.

## Error And Empty States

- If backend catalog is unavailable, training and inference controls remain visible but disabled with Chinese status copy.
- If no models exist, inference and model library explain that training can produce Pickle artifacts or users can upload a trusted Pickle model.
- If inference has failed rows, show failed status in the inference matrix and summarize failure count.

## Testing

- Update the Chinese UI structural test to assert the new tab-specific shells and Chinese labels exist.
- Add dashboard helper tests for model asset rows and inference interpretation.
- Verify browser rendering against the local frontend after implementation.
