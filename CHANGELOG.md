# Change Log
All notable changes to this project will be documented in this file.
 
The format is based on [Keep a Changelog](http://keepachangelog.com/)
and this project adheres to [Semantic Versioning](http://semver.org/).

## [0.1.3] - 2021-09-20
### Changed
- `TestSuite.run` now returns a dict instead of a list of `TestResult`
- `EvaluationResult.get_json`: results -> detail to avoid confusion

### Added
- `TestResult` now has `get_json` method to return a dict

## [0.1.2] - 2021-08-30

### Changed
- Remove unused `case_id` in `create_agent`: now `create_agent` only accepts
`**kwargs` to pass to `Agent()` as initialization params.

## [0.1.1] - 2021-08-27

### Added
- Examples on integrating grader with aivle-gym (both single-agent envs and 
multi-agent envs)


## [0.1.0] - 2021-08-26

### Added
- Initial release