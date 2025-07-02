# Contribution Guideline

Trace is an actively growing project and under active maintenance and development! We maintain two major branches `main` and `experimental`. The `main` branch is the most stable, version-controlled branch and it is what the PyPI package is linked to.  On the other hand, the `experimental` branch is the dev branch, which will change more dynamically in in preparation for the next version update. 

### Development and Review Process

Contribution to these two branches requires going through a review process via PR and passing all unit tests in CI. 
Merging a PR requires at least one reviewer different from the contributor, except for those marked as [**LIGHT**] below. 

Here is an outline: 

1. `main` will be regularly updated by PRs based on the development of the `experimental` branch following the [roadmap doc](https://docs.google.com/spreadsheets/d/1dMoECd2Soj6bATpkNDeaMxl0ymOYCtGq7ZiHr0JRdJU/edit?usp=sharing). Each update will result in a version update.

2. Except for the planned roadmap, `main` will only be updated to fix bugs.  Bug fix to what is in `main` should be submitted as PR to `main`. This will trigger a quicker review (< 3 days) and result in a version update in the third digit, and the `experimental` branch will then rebase on the updated `main`.

3. For feature development, PR should be submitted to the `experimental` branch without version update. Generally, the `experimental` branch aims to realize the milestones listed in the next version update in the [roadmap doc](https://docs.google.com/spreadsheets/d/1dMoECd2Soj6bATpkNDeaMxl0ymOYCtGq7ZiHr0JRdJU/edit?usp=sharing). If applicable, new determinstic unit tests should be added under `tests/unit_tests`, or an example run script should be added in `examples`.

4. [**LIGHT**]  Bugs fix to the new changes introduced in the `experimental` branch should be submitted as a PR to the `experimental` branch. This PR will be incoporated quickly with a light review. 

5. [**LIGHT**]  For contributions under the directory `opto/features`, they should be submitted as PR to the `experimental` branch. These usually are not under roadmap and are content not made as dependable by codes in other directories. That is, contents under `opto/features/A` should not be imported by files other than those under `opto/features/A`. So long as this rule is met, the PR will be incorprated under a light review.

6. Updates to non-coding elements (like documents) do not require a PR.

The above is applicable to all contributors, including the core contributors and maintainers.

In a regular development cycle, all features and bug fixes are merged into the experimental branch. After the items listed in the [roadmap doc](https://docs.google.com/spreadsheets/d/1dMoECd2Soj6bATpkNDeaMxl0ymOYCtGq7ZiHr0JRdJU/edit?usp=sharing) are all added to the `experimental` branch, a version branch (e.g., `v0.2.1`) will be created from `experimental`, and it will be staged for a release (to be merged into the `main` branch with a PR). At this point, the version number of the `experimental` branch will be updated to start the development of the next version.

![workflow](https://github.com/AgentOpt/Trace/blob/experimental/docs/images/contributing_workflow.png?raw=true)

### Communication

1. Quick questions should be posted on Discord channel.

2. For bugs, feature requests, contributions, or questions that might be related to a broader audience, post them as issues on the github page.

### Other Branches

In addition to `main` and `experimental`, other branches have a naming convention or `vx.x.x` for version branchs, or of `feature/xxx` or `fix/xxx`, which implements the items on the roadmap. They will be merged into the `main` or `experimental` accordingly following the rules above. Once merged, they will be removed.


# Steps for Contributions

We welcome your contributions and involvement. Below are instructions for how to contribute to Trace.

## Quick Bug Fix

If there is a minor, isolated bug that can be directly fixed, please report it as an issue or submit a PR to be merged into the `main` branch or `experimental` branch, depending on where the issue arises.


## Contributing Feature

We welcome new ideas. 

### Step 1: Create an Issue

If your changes are expected to involve in less 50 lines of codes, create an issue titled "[LIGHT] XXX". You should describe the motivation, give an overview of the change (e.g., by a pseudo-code) and its desired effects. Otherwise, create an issue titled "[MAJOR] XXX". You should write a more detailed description of your motivation, design, and demo. If more space is needed, you can attach a link to a Google doc. People should be allowed to comment on the doc.

### Step 2: Implement Feature

Create a separate branch, extending from the `experimental` branch, which contains all the new features that have not been merged into the `main` branch yet. 
Make sure your features are implemented, along with `unit tests` or `examples` to show how it's used.

### Step 3: Create a Pull Request

Create a PR formally to merge into the experiment branch and request a review. For standalone features, put the changes under `opto/features/`. This will trigger the lightest review that only checks for malicious code, or if the feature does not pass its own unit tests. For changes to the rest, expect a slightly longer review process as we work out how the changes should be integrated with the core library. Also, [LIGHT] issues can expect faster review than [MAJOR].

### Step 4: Merge into Experimental
Once the request is approved, it will be merged into the `experimental` branch.


