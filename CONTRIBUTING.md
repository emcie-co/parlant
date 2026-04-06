# DCO Sign Off

All commits must be signed off with the Developer Certificate of Origin ([DCO.md](DCO.md)).
This attests that you have the rights to submit your contribution under our project's license (Apache 2.0).

To sign off your commits:

1. Configure your Git client with your github account details:
   ```
   git config --global user.name "Your Name"
   git config --global user.email "your.email@example.com"
   ```
2. If you've configured git to use our hooks (`.githooks`), you are now ready. Otherwise, either:
   1. use our `.githooks`:
      ```
      git config set core.hookspath .githooks
      ```
      **OR**  
   2. Add the `-s` flag when committing:
      ```
      git commit -s -m "Your commit message"
      ```
### Or 

* Add the sign-off manually with:
   ```
   Signed-off-by: Your Name <your.email@example.com>
   ```