After careful analysis and reflection, I've decided **not to proceed** with this renaming.

## 💡 Decision

I will keep `pdm` focused on **polynomial dynamic models**. If in the future I implement a significant generalization to broader Bayesian dynamic models, I will create a **new `bdm` package** instead of renaming this one.

## 🎯 Reasons

- The generalization is not yet implemented (would be premature)
- `pdm` already has value as a specialized package
- Renaming would break compatibility without immediate benefit
- Creating a new package in the future offers more flexibility
- Polynomial models as "special cases" will be better handled in a separate architecture

## 📋 Next Steps

- Continue developing `pdm` focused on polynomial models
- Document long-term vision (ROADMAP.md or future issue)
- Create `bdm` only when there are concrete use cases that justify it

Thanks for the work, Copilot Agent! This PR was useful to validate the decision. 🙏

Closing without merge. Can be reopened if context changes in the future.