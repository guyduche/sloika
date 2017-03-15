# Development Cycle

### Branch off origin/master
`git checkout -b <NEW_BRANCH_NAME> origin/master`

### Edit-Commit Cycle
Go through this a few times a day, grouping changes that logically belong together in the same commit and giving them a meaningful commit message:
  * make a change: edit existing files, add new ones, etc.
  * `git status` -- inspect the changes
  * `git add <FILE_NAME>` -- add new file or stage changes in existing file
  * `git rm <FILE_NAME>` -- remove file
  * `git diff` -- inspect the diff between stage and checkout
  * `git commit` -- commit once everything you'd like to be part of this commit is staged; this will bring up an editor where commit message could be written

There are some shortcuts, e.g.
  * `git commit -a` -- stage everything git knows about and commit
  * `git commit -am"<COMMIT_MESSAGE>"` -- stage everything and commit, providing commit message in-place

### Rebase-Push Cycle

Go through this once at the end of the day or more frequently if necessary, e.g. when you want to share your changes with others, make sure your changes are backed up or get continuous integration to build your branch:
  * `git push origin <NEW_BRANCH_NAME> --set-upstream` -- if remote branch does not existyet, push the branch up, give it name that coincides with local name and start tracking it
  * `git push` -- push your changes up this way if remote branch was setup already
  * if push did not succeed because remote branch has moved on
     * `git pull --rebase` -- pull down changes from remote and apply your changes on top
     * resolve the conflicts if any (see [Conflit Resolution Cycle](#conflict-resolution-cycle) section)
     * `git push`

### Conflit Resolution Cycle

`git pull --rebase` does a fetch to get the latest changes from the remote and then rebases your changes on top. In other words, it "replays" your changes commit-by-commit as if you branched off the remote as it is at the time of pull.

For every new local commit git generates a patch and if the patch does not apply cleanly it stops and asks to resolve the conflicts. There can be more than one file that is in conflict state. File name of each file that is in conflict is printed on a separate line that starts with `CONFLICT`.

Moreover, within each file there could be more than one place where the conflicts need to be resolved. Each place starts with `<<<<<<` and ends with `>>>>>>`. Separator `=====` separates two alternative changesets that are in conflict with each other. All of these need to be resolved, and once this is done, the fact that they were all resolved in `FILE_NAME` is communicated to git like so: `git add <FILE_NAME>`.

Files that await resolution can be inspected by running `git status` which would have files with conflicts in them unstaged and listed with `both modified` suffix.

When all conflicts were resolved, git can be instructed to continue with the rebase: `git rebase --continue`. It will resume applying your work commit-by-commit and may enter the conflict state again, in which case the process will need to be repeated starting from the resolution step.
