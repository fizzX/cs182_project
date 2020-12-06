# import unittest
# from games import SecurityGame, NormalFormGame, ElectionGame
from dobbs import Dobbs
# from multipleLP import MultipleLP, Multiple_SingleLP
# from eraser import Eraser
# from origami import Origami
# from origami_milp import OrigamiMILP
# from hbgs import HBGS
import time
from games import ElectionGame, NormalFormGame, DemocratElectionGame
from eraser import Eraser
from origami import Origami




# sec_game = ElectionGame(max_coverage=40)
# p1_eraser = Eraser(sec_game)
# tic = time.perf_counter()
# p1_eraser.solve()
# toc = time.perf_counter()
# print(p1_eraser.opt_defender_payoff)
# print(p1_eraser.opt_coverage)
# print(f"Completed in {toc - tic:0.4f} seconds ")


for i in range(51):
    sec_game = ElectionGame(max_coverage=i+1)
    p1_eraser = Eraser(sec_game)
    p1_eraser.solve()
    print(p1_eraser.opt_defender_payoff)



# sec_game = ElectionGame(max_coverage=30)
# p1_origami = Origami(sec_game)
# tic = time.perf_counter()
# p1_origami.solve()
# toc = time.perf_counter()
# print(p1_origami.opt_defender_payoff)
# print(f"Completed in {toc - tic:0.4f} seconds ")

# print(p1_eraser.status)
# print(p1_eraser.opt_coverage)


# sec_game = NormalFormGame(game=sec_game,harsanyi=False)
# dobbs_game = Dobbs(sec_game)
# tic = time.perf_counter()
# dobbs_game.solve()
# toc = time.perf_counter()
# print(dobbs_game.opt_defender_mixed_strategy)
# print(dobbs_game.opt_defender_payoff)
# print(f"Completed in {toc - tic:0.4f} seconds ")
