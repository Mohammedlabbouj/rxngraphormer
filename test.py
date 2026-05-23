from rxngraphormer.rxngraphormer.eval import reaction_prediction

# retro-synthesis planning
uspto_50k_model_path = "./model_path/USPTO_50k"
pdt_smiles_lst = ['COC(=O)[C@H](CCCCN)NC(=O)Nc1cc(OC)cc(C(C)(C)C)c1O',
                  'O=C(Nc1cccc2cnccc12)c1cc([N+](a=O)[O-])c(Sc2c(Cl)cncc2Cl)s1',
                  'CCN(CC)Cc1ccc(-c2nc(C)c(COc3ccc([C@H](CC(=O)N4C(=O)OC[C@@H]4Cc4ccccc4)c4ccon4)cc3)s2)cc1']
rct_preds = reaction_prediction(uspto_50k_model_path, pdt_smiles_lst, task_type="retro-synthesis")

# forward-synthesis planning
uspto_480k_model_path = "./model_path/USPTO_480k"
rct_smiles_lst = ['C1CCOC1.CC(C)C[Mg+].CON(C)C(=O)c1ccc(O)nc1.[Cl-]',
                  'CN.O.O=C(O)c1ccc(Cl)c([N+](=O)[O-])c1',
                  'CCn1cc(C(=O)O)c(=O)c2cc(F)c(-c3ccc(N)cc3)cc21.O=CO']
pdt_preds = reaction_prediction(uspto_480k_model_path, rct_smiles_lst, task_type="forward-synthesis")