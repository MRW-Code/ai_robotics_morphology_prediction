import pandas as pd

def filter_by_solvent(df):
    """
    This function removes all the problematic cases that were seen for the most common solvents.
    If cases remain this is the function to append to deal with them.
    """
    df = df[~df['Solvent'].str.contains('/')]
    df = df[~df['Solvent'].str.contains(':')]
    df = df[~df['Solvent'].str.contains('-')]
    df = df[~df['Solvent'].str.contains('and')]
    df = df[~df['Solvent'].str.contains('or')]
    df = df[~df['Solvent'].str.contains(',')]
    return df

def solvent_syntax_correction(df, min_sol_count):
    '''
    A dirty function that manually corrects all the syntax in the solvent column.
    User should check for any additional corrections needed in custom datasets.
    min_sol_count is the minimum number of counts a solvent needs to remain in the dataset.
    e.g min_sol_count = 100 will remove any solvent not listed for >100 examples.
    '''
    df['Solvent'] = df['Solvent'].str.lower()
    df.loc[df.Solvent == 'aqueous ethanol', 'Solvent'] = 'ethanol'
    df.loc[df.Solvent == 'absolute ethanol', 'Solvent'] = 'ethanol'
    df.loc[df.Solvent == 'thf', 'Solvent'] = 'tetrahydrofuran'
    df.loc[df.Solvent == '95% ethanol', 'Solvent'] = 'ethanol'
    df.loc[df.Solvent == 'dmso', 'Solvent'] = 'dimethylsulfoxide'
    df.loc[df.Solvent == 'dimethyl sulfoxide', 'Solvent'] = 'dimethylsulfoxide'
    df.loc[df.Solvent == 'aqueous methanol', 'Solvent'] = 'methanol'
    df.loc[df.Solvent == 'absolute methanol', 'Solvent'] = 'methanol'
    df.loc[df.Solvent == 'ether', 'Solvent'] = 'diethyl ether'
    df.loc[df.Solvent == 'dmf', 'Solvent'] = 'dimethylformamide'
    df = df.groupby('Solvent').filter(lambda x: len(x) > min_sol_count)
    df = df[~df['Solvent'].str.contains('vapour deposition')]
    df = df[~df['Solvent'].str.contains('sublimation')]
    df = df[~df['Solvent'].str.contains('petroleum ether')]  # removed as no smiles possible
    df = df[~df['Solvent'].str.contains('hexanes')]  # no isomer info so removed
    print(f'Solvent cleaning applied, data contains {df["Solvent"].value_counts().shape[0]} unique solvents')
    return df

def habit_syntax_correction(df, min_habit_count):
    '''
    A dirty function that manually corrects all the syntax in the habit column.
    User should check for any additional corrections needed in custom datasets.
    min_habit_count is the minimum number of counts a habit needs to remain in the dataset.
    e.g min_habit_count = 100 will remove any habit not listed for >100 examples.
    '''
    df['Habit'] = df['Habit'].str.lower()
    df.loc[df.Habit == 'needles', 'Habit'] = 'needle'
    df.loc[df.Habit == 'plates', 'Habit'] = 'plate'
    df.loc[df.Habit == 'blocks', 'Habit'] = 'block'
    df.loc[df.Habit == 'prisms', 'Habit'] = 'prism'

    df.loc[df.Habit == 'prismatic', 'Habit'] = 'prism'
    df.loc[df.Habit == 'platelet', 'Habit'] = 'plate'
    df.loc[df.Habit == 'cube', 'Habit'] = 'block'
    df.loc[df.Habit == 'parallelepiped', 'Habit'] = 'block'
    df.loc[df.Habit == 'rod', 'Habit'] = 'needle'

    df = df.groupby('Habit').filter(lambda x: len(x) > min_habit_count)
    print(f'Habit cleaning applied, data contains {df["Habit"].value_counts().shape[0]} unique Habits')
    return df

def do_cleaning(df, min_sol_count, min_habit_count, save_output):
    df = df.dropna(axis=0)
    df = filter_by_solvent(df)
    df = habit_syntax_correction(df, min_habit_count)
    df = solvent_syntax_correction(df, min_sol_count)
    if save_output: df.to_csv('./csd/processed/clean_csd_output.csv')
    return df
