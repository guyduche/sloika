This is a description of workflow of git.
========================================
Part I: Development Cycle
========================================
Step 1: Branching off master
        
        1-a) git clone ...
	1-b) git checkout                    ensure that it is up-to-date with master branch
     	1-c) git checkout -b <branch>        branch off master

Step 2: Edit

	2-a) git status
	2-b) do your edit

Step 3: Commit

	3-a) git status
	3-b) git diff
	3-c) git commit -a

Step 4: Rebase and Push

	4-a) git push origin <branch> --set-upstream    push the branch and upstream if it is not there yet
	4-b) git push					otherwise
	4-c-1) if worked the job is done.
	4-c-2) if not then do the following:
		i)   git pull --rebase
		ii)  resolve the conflict
		iii) git push


=========================================
Part II: Resolving Conficts
=========================================
If there is a conflict between different local versions, then the follwoing steps are suggested to resolve it:

Step 1: git pull --rebase

Step 2: lookfor files in conflict.
	The conflict are between <<<<<<<< and >>>>>>> marks.

Step 3: Resolve all of them.

Step 4: When you are done with a file do
		git add <filename>
	
Step 5: To tell git you have resolved the conflict:
         	git rebase --continue
