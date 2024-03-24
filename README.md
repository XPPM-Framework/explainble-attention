# Sinhghatta2020

## Usage

## Parameters

Some parameters are listed here.

| Parameter             | Datatype | Default | Description                                                 |
|-----------------------|----------|---------|-------------------------------------------------------------|
| `incl_time`           | `bool`   | `True`  | Whether to include timestamp information.                   |
| `incl_res`            | `bool`   | `True`  | Whether to include resource information.                    |
| `perform_role_mining` | `bool`   | `True`  | Whether to perform role mining when preprocessing the data. |

## Output

Output is created in the given `model_path` in the form of an `.h5` file for the model and next to it a `.json` file for some parameters of the model.

Additionally, the method internally creates a folder under the `folder` parameter. This will default to `MY_WORKSPACE_DIR / "output_files"`.
This only contains some intermediate models and some the model parameters again. The content of this folder is not relevant to us.
