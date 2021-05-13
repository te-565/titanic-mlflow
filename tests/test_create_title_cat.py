import pandas as pd
from src.utils import (
    load_config,
    load_parameters,
)
from src.preprocessing_pipeline import create_title_cat


def test_create_title_cat():
    """Test the create_title_cat function"""

    # Load in the test configuration & parameters
    config = load_config(".env-test")
    parameters = load_parameters(parameters_path=config["parameters_path"])
    title_codes = (
        parameters["pipeline_parameters"]["create_title_cat_kw_args"]
        ["title_codes"]
    )

    data = [
        dict(id=1, col1="Tyrell, Ms. Olenna"),
        dict(id=2, col1="Baratheon, Master. Joffrey"),
        dict(id=3, col1="Lannister, Major. Tyrion"),
        dict(id=4, col1="Targaeryn, Miss. Danaerys"),
        dict(id=5, col1="Snow, Mr. Jon Eddard"),
        dict(id=6, col1="Stark, Mme. Arya"),
        dict(id=7, col1="Sparrow, Rev. High"),
        dict(id=8, col1="Lannister, Dr. Jamie"),
        dict(id=9, col1="Stark, Mrs. Catelyn"),
        dict(id=10, col1="Cooper, Major. Bronn"),
        dict(id=11, col1="Tyrell, Lady. Margaery"),
        dict(id=12, col1="Stark, Mme. Sansa"),
        dict(id=13, col1="Baratheon, Sir. Robert"),
        dict(id=14, col1="Tarth, Mlle. Brienne"),
        dict(id=15, col1="Mormont, Col. Jorah"),
        dict(id=16, col1="Clegane, Capt. Sandor"),
        dict(id=17, col1="Greyjoy, Countess. Yara"),
        dict(id=18, col1="Giantsbane, Jonkheer. Tormund"),
        dict(id=19, col1="Tarly, Dona. Samwell")
    ]

    df = pd.DataFrame(data).set_index("id", drop=True)

    # Run the function
    df_out = create_title_cat(
        df=df,
        source_column="col1",
        dest_column="col2",
        title_codes=title_codes
    )

    assert df_out["col2"].loc[1] == "gen_female"
    assert df_out["col2"].loc[2] == "young_male"
    assert df_out["col2"].loc[3] == "other_male"
    assert df_out["col2"].loc[4] == "young_female"
    assert df_out["col2"].loc[5] == "gen_male"
    assert df_out["col2"].loc[6] == "gen_female"
    assert df_out["col2"].loc[7] == "other_male"
    assert df_out["col2"].loc[8] == "other_male"
    assert df_out["col2"].loc[9] == "gen_female"
    assert df_out["col2"].loc[10] == "other_male"
    assert df_out["col2"].loc[11] == "other_female"
    assert df_out["col2"].loc[12] == "gen_female"
    assert df_out["col2"].loc[13] == "other_male"
    assert df_out["col2"].loc[14] == "young_female"
    assert df_out["col2"].loc[15] == "other_male"
    assert df_out["col2"].loc[16] == "other_male"
    assert df_out["col2"].loc[17] == "other_female"
    assert df_out["col2"].loc[18] == "other_male"
    assert df_out["col2"].loc[19] == "other_female"
