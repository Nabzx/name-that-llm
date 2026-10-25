# Speaker notes: Name That LLM

One short block per slide. About 360 words, under three minutes at a calm pace.

## 1. Cover (0:00)

> Hi, I'm Nabil. This is my entry to Name That LLM.

## 2. The task (0:05)

> Each passage was written by one of eighteen AI models. My job is to guess which one. Different companies are easy to tell apart. Two versions of the same model are much harder.

## 3. Literature review (0:18)

> Before writing any code, I read the best public notebooks and forum threads, and wrote short notes on each idea. Three lessons: more data matters more than bigger models, the best public models were free to use, and I'd need my own way to test ideas.

## 4. Starting point (0:36)

> So I set up a fair test first, keeping each prompt's passages together. A simple model scored 0.348. Then I loaded the best public models with my own code.

## 5. Approach (0:48)

> Several models each make a guess, and I combine their guesses. The black boxes are mine, including a model I trained myself overnight.

## 6. Hypotheses (0:59)

> My four ideas: writing habits give a model away, more data helps most, passages on the same prompt hide the author's style, and the test isn't evenly split between models.

## 7. Leaderboard results (1:14)

> I changed one thing at a time. I started at 0.626 and got stuck at 0.631 for a while. Then my best change took me to 0.637. I picked probes nine and seven as my final two.

## 8. Tested offline (1:31)

> These scores are on data each model never saw. The biggest gain came from comparing passages that answer the same prompt: plus 0.058.

## 9. What didn't work (1:44)

> One public trick dropped my score to 0.622. It helped weaker setups but hurt mine, so I removed it.

## 10. What the leaderboard taught me (1:55)

> Small gains often don't show on the public scores, so I trusted my own tests. And a test is only fair if the model hasn't already seen the answers.

## 11. How I used AI (2:08)

> I used Claude Code from the start, with the same setup as my other projects: clear rules, small tasks, a note before each decision and a log of every change. That let several AI sessions work at once while I made the decisions. Every session is saved in the repo.

## 12. What I'd try next (2:28)

> Next, I'd train on more data, test every model properly, and focus on the two hardest pairs. Thanks for watching.
