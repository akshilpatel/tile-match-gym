# Test cases

This file documents the test cases planned for testing, sectioned out by high-level functionality.

## Board creation

- [ ] I can create a board of different shapes, with different numbers of tile colours. The board has no existing matches. The board is a numpy integer array of correct shape.

## Making a move

- [ ] Given a pair of coordinates on the board (which specifies a move), I can check whether the move is valid.
- [ ] Given a valid move, I can switch the board coordinates.

## Refill

- [ ] If the board has no empty tiles, the board is unchanged.
- [ ] If the board has empty tiles, the board is filled up.

## Gravity

- [ ] If the board has empty tiles at the top of their columns, there is no change.
- [ ] If the board has empty tiles at the bottom, the whole column shifts down and the empties are now at the top.
- [ ] If the board has empty tiles in the middle the non-empty tiles above shift down according to the number of intermediary empty tiles.
- [ ] Given a board with a non-empty activation queue, gravity updates the coordinates for the activation queue.

## Automatch - One round of automatching. Detect,

- [ ] Given a board with no matches, automatch does not change the board and returns False.
      Given a board with one match:
- [ ] If the match is 3+ ordinary tiles in a row or column, automatch returns True and eliminates the ordinary tiles.
- [ ] If the match is 4 tiles in a row, it creates a horizontal laser tile
- [ ]

## Create Special

## Apply activation

- [ ] Ordinary tiles. No side effects
- [ ] Ordinary tiles. Side effects.
- [ ] Laser tiles. No side effects
- [ ] Cookie + cookie
- [ ] Cookie + laser
- [ ] Cookie + bomb
- [ ] Laser + bomb
- [ ] Bomb + bomb
- [ ] Laser + laser

## Get match coordinates

- [ ] Given a board with no matches, this returns an empty list.
- [ ] Given a board with just a horizontal match, this returns a horizontal match.
- [ ] Same for vertical.
- [ ] Given a board with multiple matches at same lowest level, return all matches at same level. Multiple horizontal, multiple vertical and mixed cases.

## Get lowest v match

1. No matches
2. One match.
3. Two matches at same level.
   1. 3
   2. 5
4. Two matches at diff levels.
   1. 4 and 4
5. One long match - should only return one match and not repeated matches.

## Get lowest h match

1. No matches -> Empty list, -1
2. One match. -> List of coords, lowest level
   1. Of 3
   2. Of 4
3. Three matches at diff levels.
   1. Of 5
4. Three matches at same level.
   1. 4 and 5
5. One long match - should only return one match and not repeated matches.

## Activation loop

- [ ] If the activation queue is empty. Nothing happens.
- [ ] If the activation queue is not empty, the activation queue is processed with eliminations.
- [ ] Activation loop and automatch don't hang (since they call each other.)
