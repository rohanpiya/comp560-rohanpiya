This experiment uses a simple, character-level dataset made from alphabets, commas, semicolons and newlines. Each line follows a fixed pattern: three alphabetic characters, a comma, three alphabetic characters, a semicolon, and a newline. For example, 

aaa,bbb;
ccc,ddd;
bbb,ddd;

The goal is for the model to learn that a comma behaves like a space between two tokens, and a semicolon predicts the end of a line (i.e, a newline character). This is meant to be a very simple baseline experiment whose task is easy to evaluate by the eye. If the model consistently generates outputs that follow the same pattern, then it has successfully learned the roles of , and ; from character-level information alone.