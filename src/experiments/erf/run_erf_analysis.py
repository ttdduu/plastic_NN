import os
from pathlib import Path
from src.experiments.erf.analyze_atto_erf import main as compute_erfs
from src.experiments.erf.visualize_erf import visualize_all_erfs, create_combined_figure, create_neuron_grid, f_of_radius
from src.config import BaseConfig

def extract_wandb_id(model_dir):
    """Extract the wandb run ID from the model directory path"""
    # Split path and get the directory containing 'files'
    parts = model_dir.split('/')
    for part in parts:
        if part.startswith('offline-run-'):
            return part.split('-')[-1]
    return None

def run_erf_analysis(model_dir, scotoma_radius,class_name="apple", model_is=None, logp=False):
    # Extract wandb ID and create suffix
    eth80_dir = "/home/tomasdu/repos/datasets/eth80-padded-circular/val"
    # Create config with parameters
    config = BaseConfig(parse_args=False)
    config.data.logpolar_apply = logp
    config.data.scotoma_radius = scotoma_radius
    config.data.target_class = class_name
    compute_erfs(config=config, model_dir=model_dir, eth80_dir=eth80_dir,logp=logp, model_is=model_is)

if __name__ == "__main__":

    classes = ['apple', 'car', 'cow', 'cup', 'dog', 'horse', 'pear', 'tomato']
    # Paths


    r0_nologp = '/home/tomasdu/repos/experiments/plastic_NNs/active/atto-og-mar19/WS-S/wandb/offline-run-20250321_001826-uxgl53di/files/model'
    r25_nologp = '/home/tomasdu/repos/experiments/plastic_NNs/active/atto-og-mar19/WS-S/wandb/offline-run-20250321_011627-q38nwqxc/files/model'
    r20_nologp= '/home/tomasdu/repos/experiments/plastic_NNs/active/atto-og-mar19/WS-S/wandb/offline-run-20250321_010449-1ky1vb2z/files/model'
    r15_nologp = '/home/tomasdu/repos/experiments/plastic_NNs/active/atto-og-mar19/WS-S/wandb/offline-run-20250321_005311-ch9oadtl/files/model'
    r10_nologp ='/home/tomasdu/repos/experiments/plastic_NNs/active/atto-og-mar19/WS-S/wandb/offline-run-20250321_004134-bgnahgr3/files/model'
    
    r0_logp = '/home/tomasdu/repos/experiments/plastic_NNs/active/atto-og-mar19/WS-S/wandb/offline-run-20250321_002408-gk29fe5c/files/model'
    r10_logp = '/home/tomasdu/repos/experiments/plastic_NNs/active/atto-og-mar19/WS-S/wandb/offline-run-20250321_004719-4njlzfkm/files/model'
    r15_logp = '/home/tomasdu/repos/experiments/plastic_NNs/active/atto-og-mar19/WS-S/wandb/offline-run-20250321_005857-omk9ztym/files/model'
    r20_logp = '/home/tomasdu/repos/experiments/plastic_NNs/active/atto-og-mar19/WS-S/wandb/offline-run-20250321_011035-0jywvutk/files/model'
    r25_logp = '/home/tomasdu/repos/experiments/plastic_NNs/active/atto-og-mar19/WS-S/wandb/offline-run-20250321_012214-3mu4iw5a/files/model'

    rad_model_og_atto_nologp = [[0,r0_nologp], [10,r10_nologp],[15,r15_nologp], [20,r20_nologp], [25,r25_nologp]]
    rad_model_og_atto_logp = [[15,r15_logp], [0,r0_logp], [10,r10_logp], [20,r20_logp], [25,r25_logp]]

    rad_model_lc_atto_nologp = [[0,'/home/tomasdu/repos/experiments/plastic_NNs/active/atto-lc5/WS-S/wandb/offline-run-20250320_172831-qxa61u88/files/model'],
                                [10,'/home/tomasdu/repos/experiments/plastic_NNs/active/atto-lc5/WS-S/wandb/offline-run-20250320_181429-q6zmyta1/files/model'],
                                [15, '/home/tomasdu/repos/experiments/plastic_NNs/active/atto-lc5/WS-S/wandb/offline-run-20250320_183730-ycqurqvm/files/model'],
                                [20,'/home/tomasdu/repos/experiments/plastic_NNs/active/atto-lc5/WS-S/wandb/offline-run-20250320_190030-p5h24yok/files/model'],
                                [25,'/home/tomasdu/repos/experiments/plastic_NNs/active/atto-lc5/WS-S/wandb/offline-run-20250320_192334-yu917305/files/model'],
                                ]

    rad_model_lc_atto_logp = [[0,'/home/tomasdu/repos/experiments/plastic_NNs/active/atto-lc5/WS-S/wandb/offline-run-20250320_173956-bx04pd2h/files/model'],
                               [10,'/home/tomasdu/repos/experiments/plastic_NNs/active/atto-lc5/WS-S/wandb/offline-run-20250320_182556-0p6jo8fa/files/model'],
                               [15, '/home/tomasdu/repos/experiments/plastic_NNs/active/atto-lc5/WS-S/wandb/offline-run-20250320_184858-ov04ge2d/files/model'], 
                                [20,'/home/tomasdu/repos/experiments/plastic_NNs/active/atto-lc5/WS-S/wandb/offline-run-20250320_191158-qqrwsba4/files/model'],
                                [25,'/home/tomasdu/repos/experiments/plastic_NNs/active/atto-lc5/WS-S/wandb/offline-run-20250320_193504-uhrytupk/files/model'],
                                ]

    #models_lists = [[rad_model_lc_atto_nologp,False],[rad_model_og_atto_nologp,False], [rad_model_og_atto_logp,True], [rad_model_lc_atto_logp,True]]
    #models_lists = [[rad_model_lc_atto_nologp,False],[rad_model_lc_atto_logp,True]]
    #models_lists = [[rad_model_lc_atto_nologp,False]]
    #models_lists = [[rad_model_og_atto_nologp,False]]

    ################# LH-S

    lhs_lc_nologp = [
                    [0,'/home/tomasdu/repos/experiments/plastic_NNs/active/atto-lc5/LH-S/wandb/offline-run-20250320_173414-hum5yibz/files/model'],
                    [10,'/home/tomasdu/repos/experiments/plastic_NNs/active/atto-lc5/LH-S/wandb/offline-run-20250320_182014-ig71e13e/files/model'],
                    [15,'/home/tomasdu/repos/experiments/plastic_NNs/active/atto-lc5/LH-S/wandb/offline-run-20250320_184316-pk5y9ufg/files/model'],
                    [20,'/home/tomasdu/repos/experiments/plastic_NNs/active/atto-lc5/LH-S/wandb/offline-run-20250320_190616-jnn0r8rh/files/model'],
                    [25,'/home/tomasdu/repos/experiments/plastic_NNs/active/atto-lc5/LH-S/wandb/offline-run-20250320_192922-5dzxxh27/files/model'],
    ]
    lhs_og_nologp = [
                    [0,'/home/tomasdu/repos/experiments/plastic_NNs/active/atto-og-mar19/LH-S/wandb/offline-run-20250321_002117-7r079tcp/files/model'],
                    [10,'/home/tomasdu/repos/experiments/plastic_NNs/active/atto-og-mar19/LH-S/wandb/offline-run-20250321_004428-uzun0ns4/files/model'],
                    [15,'/home/tomasdu/repos/experiments/plastic_NNs/active/atto-og-mar19/LH-S/wandb/offline-run-20250321_005605-gqemqlhd/files/model'],
                    [20,'/home/tomasdu/repos/experiments/plastic_NNs/active/atto-og-mar19/LH-S/wandb/offline-run-20250321_010744-u33rk52i/files/model'],
                    [25,'/home/tomasdu/repos/experiments/plastic_NNs/active/atto-og-mar19/LH-S/wandb/offline-run-20250321_011921-b0yhup57/files/model'],
    ]

    lc_apr8_wss = [
                    # [0, '/home/tomasdu/repos/experiments/plastic_NNs/active/atto-lc-apr8/WS-S/wandb/offline-run-20250408_182928-nzhez322/files/model'],
                    # [5, '/home/tomasdu/repos/experiments/plastic_NNs/active/atto-lc-apr8/WS-S/wandb/offline-run-20250408_184534-jscw2ie2/files/model'],
                    # [10,'/home/tomasdu/repos/experiments/plastic_NNs/active/atto-lc-apr8/WS-S/wandb/offline-run-20250408_190137-oamvro1o/files/model'],
                    # [15,'/home/tomasdu/repos/experiments/plastic_NNs/active/atto-lc-apr8/WS-S/wandb/offline-run-20250408_191741-aaha68yt/files/model'],
                    [0,'/home/tomasdu/repos/experiments/plastic_NNs/active/atto-lc-apr8/WS-S/wandb/offline-run-20250408_193345-ps00a7p6/files/model'],
                    # [21,'/home/tomasdu/repos/experiments/plastic_NNs/active/atto-lc-apr8/WS-S/wandb/offline-run-20250408_194957-iu6zyye9/files/model'],
                    # [22,'/home/tomasdu/repos/experiments/plastic_NNs/active/atto-lc-apr8/WS-S/wandb/offline-run-20250408_200612-et6t50zz/files/model'],
                    # [23,'/home/tomasdu/repos/experiments/plastic_NNs/active/atto-lc-apr8/WS-S/wandb/offline-run-20250408_202226-twjf2x9l/files/model'],
                    # [24,'/home/tomasdu/repos/experiments/plastic_NNs/active/atto-lc-apr8/WS-S/wandb/offline-run-20250408_203841-p8nwysug/files/model'],
                    # [25,'/home/tomasdu/repos/experiments/plastic_NNs/active/atto-lc-apr8/WS-S/wandb/offline-run-20250408_205500-i8ty2jwt/files/model'],
                    # [30,'/home/tomasdu/repos/experiments/plastic_NNs/active/atto-lc-apr8/WS-S/wandb/offline-run-20250408_181315-uh2g47yq/files/model'],
    ]
    lc_apr8_lhs = [
                    [0, '/home/tomasdu/repos/experiments/plastic_NNs/active/atto-lc-apr8/LH-S/wandb/offline-run-20250408_183732-n8fzbmp0/files/model'],
                    [5, '/home/tomasdu/repos/experiments/plastic_NNs/active/atto-lc-apr8/LH-S/wandb/offline-run-20250408_185337-nhjsewh4/files/model'],
                    [10,'/home/tomasdu/repos/experiments/plastic_NNs/active/atto-lc-apr8/LH-S/wandb/offline-run-20250408_190942-exa7jf7z/files/model'],
                    [15,'/home/tomasdu/repos/experiments/plastic_NNs/active/atto-lc-apr8/LH-S/wandb/offline-run-20250408_192543-yig9iuod/files/model'],
                    [20,'/home/tomasdu/repos/experiments/plastic_NNs/active/atto-lc-apr8/LH-S/wandb/offline-run-20250408_194152-ue4x5my7/files/model'],
                    [21,'/home/tomasdu/repos/experiments/plastic_NNs/active/atto-lc-apr8/LH-S/wandb/offline-run-20250408_195807-zyhfl6sn/files/model'],
                    [22,'/home/tomasdu/repos/experiments/plastic_NNs/active/atto-lc-apr8/LH-S/wandb/offline-run-20250408_201421-w8y7jul3/files/model'],
                    [23,'/home/tomasdu/repos/experiments/plastic_NNs/active/atto-lc-apr8/LH-S/wandb/offline-run-20250408_203036-tw1iv4kp/files/model'],
                    [24,'/home/tomasdu/repos/experiments/plastic_NNs/active/atto-lc-apr8/LH-S/wandb/offline-run-20250408_204654-1s0uz5t9/files/model'],
                    [25,'/home/tomasdu/repos/experiments/plastic_NNs/active/atto-lc-apr8/LH-S/wandb/offline-run-20250408_210313-yj5kpgwa/files/model'],
                    [30,'/home/tomasdu/repos/experiments/plastic_NNs/active/atto-lc-apr8/LH-S/wandb/offline-run-20250408_182124-o5gv847y/files/model'],
    ]

    og_apr9_lhs = [
                    [0,'/home/tomasdu/repos/experiments/plastic_NNs/active/atto-og-apr9/LH-S/wandb/offline-run-20250409_161241-r1qomh7f/files/model'],
                    [5,'/home/tomasdu/repos/experiments/plastic_NNs/active/atto-og-apr9/LH-S/wandb/offline-run-20250409_162014-skotps3z/files/model'],
                    [10,'/home/tomasdu/repos/experiments/plastic_NNs/active/atto-og-apr9/LH-S/wandb/offline-run-20250409_162748-28h42iy1/files/model'],
                    [15,'/home/tomasdu/repos/experiments/plastic_NNs/active/atto-og-apr9/LH-S/wandb/offline-run-20250409_163520-m44ag7o4/files/model'],
                    [20,'/home/tomasdu/repos/experiments/plastic_NNs/active/atto-og-apr9/LH-S/wandb/offline-run-20250409_164243-rwu8gz09/files/model'],
                    [21,'/home/tomasdu/repos/experiments/plastic_NNs/active/atto-og-apr9/LH-S/wandb/offline-run-20250409_165007-zq4fu66o/files/model'],
                    [22,'/home/tomasdu/repos/experiments/plastic_NNs/active/atto-og-apr9/LH-S/wandb/offline-run-20250409_165652-tlur787a/files/model'],
                    [23,'/home/tomasdu/repos/experiments/plastic_NNs/active/atto-og-apr9/LH-S/wandb/offline-run-20250409_170329-9x7sdprd/files/model'],
                    [24,'/home/tomasdu/repos/experiments/plastic_NNs/active/atto-og-apr9/LH-S/wandb/offline-run-20250409_171107-tw1mhbc2/files/model'],
                    [25,'/home/tomasdu/repos/experiments/plastic_NNs/active/atto-og-apr9/LH-S/wandb/offline-run-20250409_171846-x69ifi5h/files/model'],
                    [30,'/home/tomasdu/repos/experiments/plastic_NNs/active/atto-og-apr9/LH-S/wandb/offline-run-20250409_172621-jjois9ol/files/model'],
    ]
    og_apr9_wss = [
                    [0,'/home/tomasdu/repos/experiments/plastic_NNs/active/atto-og-apr9/WS-S/wandb/offline-run-20250409_160853-z7hirwq2/files/model'],
                    [5,'/home/tomasdu/repos/experiments/plastic_NNs/active/atto-og-apr9/WS-S/wandb/offline-run-20250409_161626-syihi52v/files/model'],
                    [10,'/home/tomasdu/repos/experiments/plastic_NNs/active/atto-og-apr9/WS-S/wandb/offline-run-20250409_162401-5y4t5xdv/files/model'],
                    [15,'/home/tomasdu/repos/experiments/plastic_NNs/active/atto-og-apr9/WS-S/wandb/offline-run-20250409_163134-z895rj3n/files/model'],
                    [20,'/home/tomasdu/repos/experiments/plastic_NNs/active/atto-og-apr9/WS-S/wandb/offline-run-20250409_163900-ld64on63/files/model'],
                    [21,'/home/tomasdu/repos/experiments/plastic_NNs/active/atto-og-apr9/WS-S/wandb/offline-run-20250409_164624-kmo0uoxj/files/model'],
                    [22,'/home/tomasdu/repos/experiments/plastic_NNs/active/atto-og-apr9/WS-S/wandb/offline-run-20250409_165349-luq143u2/files/model'],
                    [23,'/home/tomasdu/repos/experiments/plastic_NNs/active/atto-og-apr9/WS-S/wandb/offline-run-20250409_170007-zxmfkbrg/files/model'],
                    [24,'/home/tomasdu/repos/experiments/plastic_NNs/active/atto-og-apr9/WS-S/wandb/offline-run-20250409_170713-8vd6zhg4/files/model'],
                    [25,'/home/tomasdu/repos/experiments/plastic_NNs/active/atto-og-apr9/WS-S/wandb/offline-run-20250409_171457-k9z6d2tg/files/model'],
                    [30,'/home/tomasdu/repos/experiments/plastic_NNs/active/atto-og-apr9/WS-S/wandb/offline-run-20250409_172233-qpf6skuf/files/model'],
    ]
    #models_lists = [[lhs_lc_nologp,False,'lc'], [lhs_og_nologp,False,'og'],[rad_model_og_atto_nologp,False,'og'],[rad_model_lc_atto_nologp,False,'lc']]
    #models_lists = [[lhs_lc_nologp,False,'lc','LH-S'], [rad_model_lc_atto_nologp,False,'lc','WS-S'],[lhs_og_nologp,False,'og','LH-S'],[rad_model_og_atto_nologp,False,'og','WS-S']]
    #models_lists = [[lc_apr8_lhs,False,'lc','LH-S'], [lc_apr8_wss,False,'lc','WS-S']]
    models_lists = [[lc_apr8_wss,False,'lc','WS-S']]
    #models_lists = [[lc_apr8_lhs,False,'lc','LH-S']]
    #models_lists = [[og_apr9_lhs,False,'og','LH-S'], [og_apr9_wss,False,'og','WS-S']]
    should_generate_erfs = True
    should_generate_neuron_grids = False

    for models_list in models_lists: # models_lists is a list of pairs
        print(models_list)
        for pair in models_list[0]:
            
            if should_generate_erfs:
                # First generate ERF matrices for this model (original and scotoma only)
                #for target_class in classes:  # The neuron we're analyzing
                #    for input_class in classes:  # The images we're feeding
                #        print(f"\nAnalyzing {target_class} neuron with {input_class} images")
                #        print(f"Model is: {models_list[2]}")
                #        run_erf_analysis(pair[1], pair[0], target_class, model_is = models_list[2], input_class=input_class,logp=models_list[1])
                for target_class in classes:
                    run_erf_analysis(pair[1], pair[0], target_class, model_is = models_list[2], logp=models_list[1])
            
            if should_generate_neuron_grids:
                for target_class in classes:
                        create_neuron_grid(model_dir=pair[1], target_class=target_class, radius = pair[0])
                
        f_of_radius(models_list) # called models_radius_list in the f_of_radius func