# testing-pipelines

Code for the post: [Testing Data and Machine Learning Pipelines](https://eugeneyan.com/writing/testing-pipelines/) (or the additive vs. retroactive impact of new data or logic on tests).

This is my attempt to clarify my thinking on testing data and machine learning pipelines. It discusses:

- [Overview of testing scopes: unit, integration, functional, etc.](https://eugeneyan.com/writing/testing-pipelines/#smaller-testing-scopes--shorter-feedback-loops)
- [An example pipeline: behavioral logs -> batch inference output](https://eugeneyan.com/writing/testing-pipelines/#example-pipeline-behavioral-logs---inference-output)
- [Writing tests for our pipeline: unit, schema, integration](https://eugeneyan.com/writing/testing-pipelines/#implementation-tests-unit-schema-integration)
- [Adding new data (visible impressions) to our pipeline](https://eugeneyan.com/writing/testing-pipelines/#adding-new-data-or-logic--updating-pipeline-code)
- [The additive and retroactive impact of new data/logic on tests](https://eugeneyan.com/writing/testing-pipelines/#new-data-or-logic-additive-vs-retroactive-impacts)
- [A suggested lean approach to testing data and ML pipelines](https://eugeneyan.com/writing/testing-pipelines/#many-row-level-and-schema-tests-a-handful-of-the-rest)
