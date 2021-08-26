# aiVLE Grader
Auto grader for aiVLE Gym tasks.

**Disclaimer**: overall structure of this project is directly migrated from 
[aivle-runner-kit](https://github.com/edu-ai/aivle-runner-kit). I created a new repository and re-implement the existing
APIs from groundup to (1) avoid confusing names (i.e. `runner` module is defined in `runner-kit` repo, `runner` repo, a 
separate project runs the `runner` module); (2) adopt modern Python features like `abc` and rigorous type annotation;
(3) support [aivle-gym](https://github.com/edu-ai/aivle-gym).

## Getting Started

[Design details](https://pvzuww1vqx.larksuite.com/docs/docuseeHRJWAMV3p3uL7yYCOeYx)

A concrete example can be found under `./examples/aivle_gym_single.py` (you need to have 
[aivle-gym](https://github.com/edu-ai/aivle-gym) ready beforehand).