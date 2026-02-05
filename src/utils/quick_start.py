
from logging import getLogger
from itertools import product
from utils.dataset import RecDataset
from utils.dataloader import TrainDataLoader, EvalDataLoader
from utils.logger import init_logger
from utils.configurator import Config
from utils.utils import init_seed, get_model, get_trainer, dict2str
import platform
import os


def quick_start(model, dataset, config_dict, save_model=True):
    # merge config dict
    config = Config(model, dataset, config_dict) #实例化config类 “配置”
    init_logger(config)
    logger = getLogger()
    # print config infor
    logger.info('██Server: \t' + platform.node())
    logger.info('██Dir: \t' + os.getcwd() + '\n')
    logger.info(config)

    # 数据集加载
    dataset = RecDataset(config)
    # print dataset statistics
    logger.info(str(dataset))

    # 数据集划分
    train_dataset, valid_dataset, test_dataset = dataset.split()
    logger.info('\n====Training====\n' + str(train_dataset))
    logger.info('\n====Validation====\n' + str(valid_dataset))
    logger.info('\n====Testing====\n' + str(test_dataset))

    # wrap into dataloader
    train_data = TrainDataLoader(config, train_dataset, batch_size=config['train_batch_size'], shuffle=True)
    (valid_data, test_data) = (
        EvalDataLoader(config, valid_dataset, additional_dataset=train_dataset, batch_size=config['eval_batch_size']),
        EvalDataLoader(config, test_dataset, additional_dataset=train_dataset, batch_size=config['eval_batch_size']))

    ############ Dataset loadded, run model
    hyper_ret = []
    val_metric = config['valid_metric'].lower() #recall@20
    best_test_value = 0.0
    idx = best_test_idx = 0

    logger.info('\n\n=================================\n\n')

    # hyper-parameters
    hyper_ls = []
    if "seed" not in config['hyper_parameters']:
        config['hyper_parameters'] = ['seed'] + config['hyper_parameters']
    #print(config['hyper_parameters']) ['dropout', 'reg_weight', 'seed'] 包含超参数名称的列表
    for i in config['hyper_parameters']:
        hyper_ls.append(config[i] or [None])
    #print("test:", hyper_ls) [[0.8, 0.9], [0.0, 1e-05, 0.0001, 0.001], [999]]

    # combinations 组合超参数
    combinators = list(product(*hyper_ls))
    #print(combinators) [(0.8, 0.0, 999), (0.8, 1e-05, 999), (0.8, 0.0001, 999), (0.8, 0.001, 999), (0.9, 0.0, 999), (0.9, 1e-05, 999), (0.9, 0.0001, 999), (0.9, 0.001, 999)]
    total_loops = len(combinators) #循环次数

    for hyper_tuple in combinators: #取一组超参数
        # random seed reset
        for j, k in zip(config['hyper_parameters'], hyper_tuple):
            config[j] = k   #将超参数的值k赋给相应的键j
        init_seed(config['seed'])

        logger.info('========={}/{}: Parameters:{}={}======='.format(
            idx+1, total_loops, config['hyper_parameters'], hyper_tuple))

        # set random state of dataloader #pre
        train_data.pretrain_setup()  #train_data is the class from TrainDataLoader ——打乱itemID序列
        # model loading and initialization #freedom
        model = get_model(config['model'])(config, train_data).to(config['device']) #get_model
        print(model)
        logger.info(model)

        # trainer loading and initialization // 导入trainer类
        trainer = get_trainer()(config, model) ###trainer from common.trainer.py
        # debug
        # model training
        best_valid_score, best_valid_result, best_test_upon_valid = trainer.fit(train_data, valid_data=valid_data, test_data=test_data, saved=save_model)
        #########
        hyper_ret.append((hyper_tuple, best_valid_result, best_test_upon_valid))

        # save best test
        if best_test_upon_valid[val_metric] > best_test_value:
            best_test_value = best_test_upon_valid[val_metric]
            best_test_idx = idx
        idx += 1

        logger.info('best valid result: {}'.format(dict2str(best_valid_result)))
        logger.info('test result: {}'.format(dict2str(best_test_upon_valid)))
        logger.info('████Current BEST████:\nParameters: {}={},\n'
                    'Valid: {},\nTest: {}\n\n\n'.format(config['hyper_parameters'],
            hyper_ret[best_test_idx][0], dict2str(hyper_ret[best_test_idx][1]), dict2str(hyper_ret[best_test_idx][2])))

    # log info
    logger.info('\n============All Over=====================')
    for (p, k, v) in hyper_ret:
        logger.info('Parameters: {}={},\n best valid: {},\n best test: {}'.format(config['hyper_parameters'],
                                                                                  p, dict2str(k), dict2str(v)))

    logger.info('\n\n█████████████ BEST ████████████████')
    logger.info('\tParameters: {}={},\nValid: {},\nTest: {}\n\n'.format(config['hyper_parameters'],
                                                                   hyper_ret[best_test_idx][0],
                                                                   dict2str(hyper_ret[best_test_idx][1]),
                                                                   dict2str(hyper_ret[best_test_idx][2])))

