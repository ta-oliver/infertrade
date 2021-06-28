# Contributing to InferTrade

Found a new feature, or a bug? We welcome your pull requests.

## Licencing

infertrade is an [Apache 2.0 project](https://github.com/ta-oliver/infertrade/blob/main/LICENSE).

Contributions should be consistent with the [Apache 2.0 licence](https://www.apache.org/licenses/LICENSE-2.0) and with the principles of the [Developer Certificate of Origin](https://developercertificate.org/). 


## Contribution process

1. Submit an issue describing your proposed change to the repo in question.
1. The repo owner will respond to your issue promptly and the community can provide feedback.
1. If your proposed change is accepted, fork infertrade.
    ```
   git@github.com:ta-oliver/infertrade.git
    ```
1. Set up developer mode.
    
    ```
   cd infertrade/
   make dev-venv
   source .inferlib_venv/bin/activate
    ```
1. Make changes, and test your code changes. You may use `make` for testing your code.
      - Run tests:
         ```
         make test
         ```
      - Check code coverage:
         ```
         make coverage
         ```

1. Ensure that your code adheres to the existing style within InferTrade. Refer to the 
   [Google Style Guide](https://google.github.io/styleguide/pyguide.html) if unsure. We recommend you use
   ```
   make autoformat
   ```
   This will lint with the `black` package with 120 char lines setting (`black -l 120 infertrade`).
1. Ensure that your code has an appropriate set of unit tests which all pass.
1. If this is your first pull request, please add yourself to the [list of project copyright contributors](https://github.com/ta-oliver/infertrade/blob/main/docs/list_of_copyright_contributors.md).
1. Submit a pull request.
