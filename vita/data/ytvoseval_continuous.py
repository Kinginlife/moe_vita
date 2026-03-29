__author__ = 'ychfan'

import numpy as np
import datetime
import time
from collections import defaultdict
from pycocotools import mask as maskUtils
import copy

class YTVOSeval:
    # Interface for evaluating video instance segmentation on the YouTubeVIS dataset.
    #
    # The usage for YTVOSeval is as follows:
    #  cocoGt=..., cocoDt=...       # load dataset and results
    #  E = YTVOSeval(cocoGt,cocoDt); # initialize YTVOSeval object
    #  E.params.recThrs = ...;      # set parameters as desired
    #  E.evaluate();                # run per image evaluation
    #  E.accumulate();              # accumulate per image results
    #  E.summarize();               # display summary metrics of results
    # For example usage see evalDemo.m and http://mscoco.org/.
    #
    # The evaluation parameters are as follows (defaults in brackets):
    #  imgIds     - [all] N img ids to use for evaluation
    #  catIds     - [all] K cat ids to use for evaluation
    #  iouThrs    - [.5:.05:.95] T=10 IoU thresholds for evaluation
    #  recThrs    - [0:.01:1] R=101 recall thresholds for evaluation
    #  areaRng    - [...] A=4 object area ranges for evaluation
    #  maxDets    - [1 10 100] M=3 thresholds on max detections per image
    #  iouType    - ['segm'] set iouType to 'segm', 'bbox' or 'keypoints'
    #  iouType replaced the now DEPRECATED useSegm parameter.
    #  useCats    - [1] if true use category labels for evaluation
    # Note: if useCats=0 category labels are ignored as in proposal scoring.
    # Note: multiple areaRngs [Ax2] and maxDets [Mx1] can be specified.
    #
    # evaluate(): evaluates detections on every image and every category and
    # concats the results into the "evalImgs" with fields:
    #  dtIds      - [1xD] id for each of the D detections (dt)
    #  gtIds      - [1xG] id for each of the G ground truths (gt)
    #  dtMatches  - [TxD] matching gt id at each IoU or 0
    #  gtMatches  - [TxG] matching dt id at each IoU or 0
    #  dtScores   - [1xD] confidence of each dt
    #  gtIgnore   - [1xG] ignore flag for each gt
    #  dtIgnore   - [TxD] ignore flag for each dt at each IoU
    #
    # accumulate(): accumulates the per-image, per-category evaluation
    # results in "evalImgs" into the dictionary "eval" with fields:
    #  params     - parameters used for evaluation
    #  date       - date evaluation was performed
    #  counts     - [T,R,K,A,M] parameter dimensions (see above)
    #  precision  - [TxRxKxAxM] precision for every evaluation setting
    #  recall     - [TxKxAxM] max recall for every evaluation setting
    # Note: precision and recall==-1 for settings with no gt objects.
    #
    # See also coco, mask, pycocoDemo, pycocoEvalDemo
    #
    # Microsoft COCO Toolbox.      version 2.0
    # Data, paper, and tutorials available at:  http://mscoco.org/
    # Code written by Piotr Dollar and Tsung-Yi Lin, 2015.
    # Licensed under the Simplified BSD License [see coco/license.txt]
    def __init__(self, cocoGt=None, cocoDt=None, iouType='segm'):
        '''
        Initialize CocoEval using coco APIs for gt and dt
        :param cocoGt: coco object with ground truth annotations
        :param cocoDt: coco object with detection results
        :return: None
        '''
        if not iouType:
            print('iouType not specified. use default iouType segm')
        self.cocoGt   = cocoGt              # ground truth COCO API
        self.cocoDt   = cocoDt              # detections COCO API
        self.params   = {}                  # evaluation parameters
        self.evalVids = defaultdict(list)   # per-image per-category evaluation results [KxAxI] elements
        self.eval     = {}                  # accumulated evaluation results
        self._gts = defaultdict(list)       # gt for evaluation
        self._dts = defaultdict(list)       # dt for evaluation
        self.params = Params(iouType=iouType) # parameters
        self._paramsEval = {}               # parameters for evaluation
        self.stats = []                     # result summarization
        self.ious = {}                      # ious between all gts and dts
        if not cocoGt is None:
            self.params.vidIds = sorted(cocoGt.vidid_filter)  #########################################
            self.params.catIds = sorted(cocoGt.catsIds_filter)  #########################################


    def _prepare(self):
        '''
        Prepare ._gts and ._dts for evaluation based on params
        :return: None
        '''
        def _toMask(anns, coco):
            # modify ann['segmentation'] by reference
            for ann in anns:
                for i, a in enumerate(ann['segmentations']):
                    if a:
                        rle = coco.annToRLE(ann, i)
                        ann['segmentations'][i] = rle
                l = [a for a in ann['areas'] if a]
                if len(l)==0:
                  ann['avg_area'] = 0
                else:
                  ann['avg_area'] = np.array(l).mean() 
        p = self.params
        if p.useCats:
            gts=self.cocoGt.loadAnns(self.cocoGt.getAnnIds(vidIds=p.vidIds, catIds=p.catIds))
            #print("ANNIDS", self.cocoGt.getAnnIds(vidIds=p.vidIds, catIds=p.catIds), len(self.cocoGt.getAnnIds(vidIds=p.vidIds, catIds=p.catIds))) ########################
            dts=self.cocoDt.loadAnns(self.cocoDt.getAnnIds(vidIds=p.vidIds, catIds=p.catIds))
        else:
            gts=self.cocoGt.loadAnns(self.cocoGt.getAnnIds(vidIds=p.vidIds))
            dts=self.cocoDt.loadAnns(self.cocoDt.getAnnIds(vidIds=p.vidIds))

        # convert ground truth to mask if iouType == 'segm'
        if p.iouType == 'segm':
            _toMask(gts, self.cocoGt)
            _toMask(dts, self.cocoDt)
        # set ignore flag
        for gt in gts:
            gt['ignore'] = gt['ignore'] if 'ignore' in gt else 0
            gt['ignore'] = 'iscrowd' in gt and gt['iscrowd']
            if p.iouType == 'keypoints':
                gt['ignore'] = (gt['num_keypoints'] == 0) or gt['ignore']
        self._gts = defaultdict(list)       # gt for evaluation
        self._dts = defaultdict(list)       # dt for evaluation
        for gt in gts:
            self._gts[gt['video_id'], gt['category_id']].append(gt)
        for dt in dts:
            self._dts[dt['video_id'], dt['category_id']].append(dt)
        self.evalVids = defaultdict(list)   # per-image per-category evaluation results
        self.eval     = {}                  # accumulated evaluation results

    def evaluate(self):#遍历每一个视频和每一个类别，对于每个类别，它会获取所有的真实标注（GT）和预测结果（Detections），计算所有 GT 和预测之间的 IoU矩阵，执行匹配操作，为每个 GT 找到最佳的预测匹配
        '''
        Run per image evaluation on given images and store results (a list of dict) in self.evalVids
        :return: None
        '''
        tic = time.time()
        print('Running per image evaluation...')
        p = self.params
        # add backward compatibility if useSegm is specified in params
        if not p.useSegm is None:
            p.iouType = 'segm' if p.useSegm == 1 else 'bbox'
            print('useSegm (deprecated) is not None. Running {} evaluation'.format(p.iouType))
        print('Evaluate annotation type *{}*'.format(p.iouType))
        p.vidIds = list(np.unique(p.vidIds))
        if p.useCats:
            p.catIds = list(np.unique(p.catIds))#去重并标准化类别ID列表
        p.maxDets = sorted(p.maxDets) # 将可用的最大检测数阈值排序
        self.params=p

        self._prepare()# 预处理：构建 GT/DT 索引、过滤空元素、按参数裁剪数据等，便于后续快速查询
        # loop through images, area range, max detection number
        catIds = p.catIds if p.useCats else [-1] # 决定按哪些类别循环；不用类别时用占位 -1

        if p.iouType == 'segm' or p.iouType == 'bbox':
            computeIoU = self.computeIoU
        elif p.iouType == 'keypoints':
            computeIoU = self.computeOks
        self.ious = {(vidId, catId): computeIoU(vidId, catId) \
                        for vidId in p.vidIds
                        for catId in catIds} # 预先为每个 (视频, 类别) 计算 IoU/OKS 矩阵并缓存

        evaluateVid = self.evaluateVid #单个 (视频, 类别, 面积范围, 最大检测数) 的评估函数
        maxDet = p.maxDets[-1] # 取最大的检测上限（例如 100），用于实际匹配评估
        
        
        self.evalImgs = [evaluateVid(vidId, catId, areaRng, maxDet)
                 for catId in catIds# 先按类别
                 for areaRng in p.areaRng# 再按面积范围（例如小/中/大物体）
                 for vidId in p.vidIds  # 再按视频
             ] # 评估内容包括排序、阈值下匹配 TPs/FPs 的标记、置信度等，为后续 accumulate/summarize 做准备
        self._paramsEval = copy.deepcopy(self.params)
        toc = time.time()
        print('DONE (t={:0.2f}s).'.format(toc-tic))

    def computeIoU(self, vidId, catId):
        p = self.params
        if p.useCats:
            gt = self._gts[vidId,catId]
            dt = self._dts[vidId,catId]
        else:
            gt = [_ for cId in p.catIds for _ in self._gts[vidId,cId]]
            dt = [_ for cId in p.catIds for _ in self._dts[vidId,cId]]
        if len(gt) == 0 and len(dt) ==0:
            return []
        inds = np.argsort([-d['score'] for d in dt], kind='mergesort')
        dt = [dt[i] for i in inds]
        if len(dt) > p.maxDets[-1]:
            dt=dt[0:p.maxDets[-1]]

        if p.iouType == 'segm':
            g = [g['segmentations'] for g in gt]
            d = [d['segmentations'] for d in dt]
        elif p.iouType == 'bbox':
            g = [g['bboxes'] for g in gt]
            d = [d['bboxes'] for d in dt]
        else:
            raise Exception('unknown iouType for iou computation')

        # compute iou between each dt and gt region
        iscrowd = [int(o['iscrowd']) for o in gt]
        #ious = maskUtils.iou(d,g,iscrowd)
        def iou_seq(d_seq, g_seq):
            i = .0
            u = .0
            for d, g in zip(d_seq, g_seq):
                if d and g:
                    i += maskUtils.area(maskUtils.merge([d, g], True))
                    u += maskUtils.area(maskUtils.merge([d, g], False))
                elif not d and g:
                    u += maskUtils.area(g)
                elif d and not g:
                    u += maskUtils.area(d)
            if not u > .0:
                print("Mask sizes in video {} and category {} may not match!".format(vidId, catId))
            iou = i / u if u > .0 else .0
            return iou
        ious = np.zeros([len(d), len(g)])
        for i, j in np.ndindex(ious.shape):
            ious[i, j] = iou_seq(d[i], g[j])
        #print(vidId, catId, ious.shape, ious)
        return ious

    def computeOks(self, imgId, catId):
        p = self.params
        # dimention here should be Nxm
        gts = self._gts[imgId, catId]
        dts = self._dts[imgId, catId]
        inds = np.argsort([-d['score'] for d in dts], kind='mergesort')
        dts = [dts[i] for i in inds]
        if len(dts) > p.maxDets[-1]:
            dts = dts[0:p.maxDets[-1]]
        # if len(gts) == 0 and len(dts) == 0:
        if len(gts) == 0 or len(dts) == 0:
            return []
        ious = np.zeros((len(dts), len(gts)))
        sigmas = np.array([.26, .25, .25, .35, .35, .79, .79, .72, .72, .62,.62, 1.07, 1.07, .87, .87, .89, .89])/10.0
        vars = (sigmas * 2)**2
        k = len(sigmas)
        # compute oks between each detection and ground truth object
        for j, gt in enumerate(gts):
            # create bounds for ignore regions(double the gt bbox)
            g = np.array(gt['keypoints'])
            xg = g[0::3]; yg = g[1::3]; vg = g[2::3]
            k1 = np.count_nonzero(vg > 0)
            bb = gt['bbox']
            x0 = bb[0] - bb[2]; x1 = bb[0] + bb[2] * 2
            y0 = bb[1] - bb[3]; y1 = bb[1] + bb[3] * 2
            for i, dt in enumerate(dts):
                d = np.array(dt['keypoints'])
                xd = d[0::3]; yd = d[1::3]
                if k1>0:
                    # measure the per-keypoint distance if keypoints visible
                    dx = xd - xg
                    dy = yd - yg
                else:
                    # measure minimum distance to keypoints in (x0,y0) & (x1,y1)
                    z = np.zeros((k))
                    dx = np.max((z, x0-xd),axis=0)+np.max((z, xd-x1),axis=0)
                    dy = np.max((z, y0-yd),axis=0)+np.max((z, yd-y1),axis=0)
                e = (dx**2 + dy**2) / vars / (gt['avg_area']+np.spacing(1)) / 2
                if k1 > 0:
                    e=e[vg > 0]
                ious[i, j] = np.sum(np.exp(-e)) / e.shape[0]
        return ious

    def evaluateVid(self, vidId, catId, aRng, maxDet):#遍历单个视频和单个类别的所有检测结果，计算 IoU/OKS 矩阵，为每个 GT 找到最佳的预测匹配
        '''
        perform evaluation for single category and image
        :return: dict (single image results)
        '''
        p = self.params
        if p.useCats:
            gt = self._gts[vidId,catId]
            dt = self._dts[vidId,catId]
        else:
            gt = [_ for cId in p.catIds for _ in self._gts[vidId,cId]]
            dt = [_ for cId in p.catIds for _ in self._dts[vidId,cId]]
        if len(gt) == 0 and len(dt) ==0:
            return None

        for g in gt:
            if g['ignore'] or (g['avg_area']<aRng[0] or g['avg_area']>aRng[1]):
                g['_ignore'] = 1
            else:
                g['_ignore'] = 0

        # sort dt highest score first, sort gt ignore last
        gtind = np.argsort([g['_ignore'] for g in gt], kind='mergesort')
        gt = [gt[i] for i in gtind]
        dtind = np.argsort([-d['score'] for d in dt], kind='mergesort')
        dt = [dt[i] for i in dtind[0:maxDet]]
        iscrowd = [int(o['iscrowd']) for o in gt]
        # load computed ious
        ious = self.ious[vidId, catId][:, gtind] if len(self.ious[vidId, catId]) > 0 else self.ious[vidId, catId]

        T = len(p.iouThrs)
        G = len(gt)
        D = len(dt)
        gtm  = np.zeros((T,G))
        dtm  = np.zeros((T,D))
        gtIg = np.array([g['_ignore'] for g in gt])
        dtIg = np.zeros((T,D))
        if not len(ious)==0:
            for tind, t in enumerate(p.iouThrs):
                for dind, d in enumerate(dt):
                    # information about best match so far (m=-1 -> unmatched)
                    iou = min([t,1-1e-10])
                    m   = -1
                    for gind, g in enumerate(gt):
                        # if this gt already matched, and not a crowd, continue
                        if gtm[tind,gind]>0 and not iscrowd[gind]:
                            continue
                        # if dt matched to reg gt, and on ignore gt, stop
                        if m>-1 and gtIg[m]==0 and gtIg[gind]==1:
                            break
                        # continue to next gt unless better match made
                        if ious[dind,gind] < iou:
                            continue
                        # if match successful and best so far, store appropriately
                        iou=ious[dind,gind]
                        m=gind
                    # if match made store id of match for both dt and gt
                    if m ==-1:
                        continue
                    dtIg[tind,dind] = gtIg[m]
                    dtm[tind,dind]  = gt[m]['id']
                    gtm[tind,m]     = d['id']
        # set unmatched detections outside of area range to ignore
        a = np.array([d['avg_area']<aRng[0] or d['avg_area']>aRng[1] for d in dt]).reshape((1, len(dt)))
        dtIg = np.logical_or(dtIg, np.logical_and(dtm==0, np.repeat(a,T,0)))
        # store results for given image and category
        return {
                'video_id':     vidId,
                'category_id':  catId,
                'aRng':         aRng,
                'maxDet':       maxDet,
                'dtIds':        [d['id'] for d in dt],
                'gtIds':        [g['id'] for g in gt],
                'dtMatches':    dtm,
                'gtMatches':    gtm,
                'dtScores':     [d['score'] for d in dt],
                'gtIgnore':     gtIg,
                'dtIgnore':     dtIg,
            }

    def accumulate(self, p = None):#遍历多个 IoU 阈值，根据 evaluate() 的匹配结果累加 True Positives (TP) 和 False Positives (FP) 的数量
        '''
        Accumulate per image evaluation results and store the result in self.eval
        :param p: input params for evaluation
        :return: None
        '''
        print('Accumulating evaluation results...')
        tic = time.time()
        if not self.evalImgs:
            print('Please run evaluate() first')
        # allows input customized parameters
        if p is None:
            p = self.params
        p.catIds = p.catIds if p.useCats == 1 else [-1]
        T           = len(p.iouThrs)# IoU 阈值个数（默认 10 个：0.5:0.95）
        R           = len(p.recThrs)# 召回率采样点个数（默认 101 个：0:0.01:1）
        K           = len(p.catIds) if p.useCats else 1 # 类别数
        A           = len(p.areaRng) #面积分段数（all/s/m/l）
        M           = len(p.maxDets)# 最大检测数设置数（默认 3 个：1/10/100
        precision   = -np.ones((T,R,K,A,M)) # -1 for the precision of absent categories  精确率张量，初始化为 -1 表示该设置下无 gt
        recall      = -np.ones((T,K,A,M)) # 召回率（每个设置的最大召回）
        scores      = -np.ones((T,R,K,A,M))# 与 precision 对应的分数（用于绘制曲线）

        # create dictionary for future indexing
        _pe = self._paramsEval
        catIds = _pe.catIds if _pe.useCats else [-1] 
        setK = set(catIds)  # 允许的类别集合
        setA = set(map(tuple, _pe.areaRng)) # 允许的面积范围集合
        setM = set(_pe.maxDets) # 允许的最大检测数集合
        setI = set(_pe.vidIds) # 允许的视频集合
        # get inds to evaluate
        k_list = [n for n, k in enumerate(p.catIds)  if k in setK]# 映射到 _pe 的类别索引
        m_list = [m for n, m in enumerate(p.maxDets) if m in setM]# 映射到 _pe 的 maxDet 值（直接是值）
        a_list = [n for n, a in enumerate(map(lambda x: tuple(x), p.areaRng)) if a in setA] # 面积分段索引
        i_list = [n for n, i in enumerate(p.vidIds)  if i in setI] # 视频索引
        I0 = len(_pe.vidIds)# evaluate() 中视频数量
        A0 = len(_pe.areaRng)  # evaluate() 中面积分段数量
        # retrieve E at each category, area range, and max number of detections
        for k, k0 in enumerate(k_list): # 遍历类别维
            Nk = k0*A0*I0 # 该类别在 evalImgs 列表中的起始步长
            for a, a0 in enumerate(a_list):# 遍历面积分段
                Na = a0*I0    # 该面积分段在 evalImgs 中的偏移步长
                for m, maxDet in enumerate(m_list):# 遍历不同的最大检测数设置
                    E = [self.evalImgs[Nk + Na + i] for i in i_list] # 取出该 (k,a) 下所有视频的评估结果条目
                    E = [e for e in E if not e is None]   # 过滤空结果（无 gt 与无 dt）
                    if len(E) == 0:
                        continue
                    dtScores = np.concatenate([e['dtScores'][0:maxDet] for e in E]) # 拼接并截取每视频前 maxDet 个检测分数

                    # different sorting method generates slightly different results.
                    # mergesort is used to be consistent as Matlab implementation.
                    inds = np.argsort(-dtScores, kind='mergesort')# 统一用稳定排序
                    dtScoresSorted = dtScores[inds] # 分数降序

                    dtm  = np.concatenate([e['dtMatches'][:,0:maxDet] for e in E], axis=1)[:,inds]# 按排序重排匹配矩阵（TP:>0）
                    dtIg = np.concatenate([e['dtIgnore'][:,0:maxDet]  for e in E], axis=1)[:,inds]# 重排忽略标记
                    gtIg = np.concatenate([e['gtIgnore'] for e in E])    # GT 忽略标记
                    npig = np.count_nonzero(gtIg==0 ) # 有效 GT 数量
                    if npig == 0:
                        continue
                    tps = np.logical_and(               dtm,  np.logical_not(dtIg) )# 真阳性：匹配且不忽略
                    fps = np.logical_and(np.logical_not(dtm), np.logical_not(dtIg) )  # 假阳性：未匹配且不忽略

                    tp_sum = np.cumsum(tps, axis=1).astype(dtype=np.float) # 按检测数量累积 TP
                    fp_sum = np.cumsum(fps, axis=1).astype(dtype=np.float) # 按检测数量累积 FP
                    for t, (tp, fp) in enumerate(zip(tp_sum, fp_sum)):# 针对每个 IoU 阈值一条 PR 曲线
                        tp = np.array(tp)
                        fp = np.array(fp)
                        nd = len(tp) # 排序后的检测数
                        rc = tp / npig # 召回率
                        pr = tp / (fp+tp+np.spacing(1)) # 精确率
                        q  = np.zeros((R,)) # 采样到固定 R 个召回点后的精确率
                        ss = np.zeros((R,)) # 同位置的分数

                        if nd:
                            recall[t,k,a,m] = rc[-1] # # 该设置下的最大召回
                        else:
                            recall[t,k,a,m] = 0

                        # numpy is slow without cython optimization for accessing elements
                        # use python array gets significant speed improvement
                        pr = pr.tolist(); q = q.tolist()

                        for i in range(nd-1, 0, -1): # 插值单调化：从右向左做 upper envelope，保证精确率非增
                            if pr[i] > pr[i-1]:
                                pr[i-1] = pr[i]
                        # 在固定的 R 个召回点上取值
                        inds = np.searchsorted(rc, p.recThrs, side='left') # 找到每个召回阈值应取的索引位置
                        try:
                            for ri, pi in enumerate(inds):
                                q[ri] = pr[pi]# 该召回点对应的精确率
                                ss[ri] = dtScoresSorted[pi]# 该点对应的分数
                        except:
                            pass
                        precision[t,:,k,a,m] = np.array(q)# 写入该 IoU 阈值的整条精确率曲线
                        scores[t,:,k,a,m] = np.array(ss) # 写入对应分数曲线
        self.eval = {
            'params': p,# 使用的参数
            'counts': [T, R, K, A, M], # 各维度大小
            'date': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), # 时间戳
            'precision': precision, # [T x R x K x A x M] 精确率张量
            'recall':   recall,# [T x K x A x M] 最大召回
            'scores': scores,# [T x R x K x A x M] 分数（
        }
        toc = time.time()
        print('DONE (t={:0.2f}s).'.format( toc-tic))

    def summarize(self):#接收 accumulate() 生成的统计数据计算出最终的指标，
        '''
        Compute and display summary metrics for evaluation results.
        Note this functin can *only* be applied on the default parameter setting
        '''
        def _summarize( ap=1, iouThr=None, areaRng='all', maxDets=100 ):
            p = self.params
            iStr = ' {:<18} {} @[ IoU={:<9} | area={:>6s} | maxDets={:>3d} ] = {:0.6f}'  ########## 0.3
            # 打印格式：指标名(AP/AR)、IoU阈值范围或单阈值、面积分段(all/s/m/l)、最大检测数、均值
            titleStr = 'Average Precision' if ap == 1 else 'Average Recall'
            typeStr = '(AP)' if ap==1 else '(AR)'
            iouStr = '{:0.2f}:{:0.2f}'.format(p.iouThrs[0], p.iouThrs[-1]) \
                if iouThr is None else '{:0.2f}'.format(iouThr) # IoU 显示：区间或单阈值

            aind = [i for i, aRng in enumerate(p.areaRngLbl) if aRng == areaRng]# 面积分段下标（如 'all'/'small'...）
            mind = [i for i, mDet in enumerate(p.maxDets) if mDet == maxDets] #最大检测数下标（对应 1/10/100 等）
            if ap == 1:
                # dimension of precision: [TxRxKxAxM]
                s = self.eval['precision'] # 取精确率张量
                # IoU
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]# 找到指定 IoU 阈值的索引
                    s = s[t] # 切片仅该阈值
                s = s[:,:,:,aind,mind]# 选择指定的面积与 maxDet 切片（保留 T/R/K 维）
            else:
                # dimension of recall: [TxKxAxM]
                s = self.eval['recall']
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                s = s[:,:,aind,mind]
            if len(s[s>-1])==0:# -1 表示该设置下无有效样本
                mean_s = -1
            else:
                mean_s = np.mean(s[s>-1])# 对有效元素取平均作为指标
            print(iStr.format(titleStr, typeStr, iouStr, areaRng, maxDets, mean_s))
            return mean_s
        def _summarizeDets():
            stats = np.zeros((12,))
            stats[0] = _summarize(1)# AP @[.50:.95] (all, maxDets=默认100)
            stats[1] = _summarize(1, iouThr=.5, maxDets=self.params.maxDets[2]) # AP50
            stats[2] = _summarize(1, iouThr=.75, maxDets=self.params.maxDets[2]) # AP75
            stats[3] = _summarize(1, areaRng='small', maxDets=self.params.maxDets[2])
            stats[4] = _summarize(1, areaRng='medium', maxDets=self.params.maxDets[2])
            stats[5] = _summarize(1, areaRng='large', maxDets=self.params.maxDets[2])
            stats[6] = _summarize(0, maxDets=self.params.maxDets[0])# AR @maxDets=1
            stats[7] = _summarize(0, maxDets=self.params.maxDets[1]) # AR @maxDets=10
            stats[8] = _summarize(0, maxDets=self.params.maxDets[2])# AR @maxDets=100
            stats[9] = _summarize(0, areaRng='small', maxDets=self.params.maxDets[2])
            stats[10] = _summarize(0, areaRng='medium', maxDets=self.params.maxDets[2])
            stats[11] = _summarize(0, areaRng='large', maxDets=self.params.maxDets[2])
            return stats
        def _summarizeKps():
            stats = np.zeros((10,))
            stats[0] = _summarize(1, maxDets=20)
            stats[1] = _summarize(1, maxDets=20, iouThr=.5)
            stats[2] = _summarize(1, maxDets=20, iouThr=.75)
            stats[3] = _summarize(1, maxDets=20, areaRng='medium')
            stats[4] = _summarize(1, maxDets=20, areaRng='large')
            stats[5] = _summarize(0, maxDets=20)
            stats[6] = _summarize(0, maxDets=20, iouThr=.5)
            stats[7] = _summarize(0, maxDets=20, iouThr=.75)
            stats[8] = _summarize(0, maxDets=20, areaRng='medium')
            stats[9] = _summarize(0, maxDets=20, areaRng='large')
            return stats
        if not self.eval:
            raise Exception('Please run accumulate() first')
        iouType = self.params.iouType
        if iouType == 'segm' or iouType == 'bbox':
            summarize = _summarizeDets
        elif iouType == 'keypoints':
            summarize = _summarizeKps
        self.stats = summarize()

    def __str__(self):
        self.summarize()

class Params:
    '''
    Params for coco evaluation api
    '''
    def setDetParams(self):
        self.vidIds = []
        self.catIds = []
        # np.arange causes trouble.  the data point on arange is slightly larger than the true value
        #self.iouThrs = np.linspace(.5, 0.95, np.round((0.95 - .5) / .05) + 1, endpoint=True)
        #self.recThrs = np.linspace(.0, 1.00, np.round((1.00 - .0) / .01) + 1, endpoint=True)
        self.iouThrs = np.linspace(.5, 0.95, int(np.round((0.95 - .5) / .05)) + 1, endpoint=True)
        self.recThrs = np.linspace(.0, 1.00, int(np.round((1.00 - .0) / .01)) + 1, endpoint=True)
        self.maxDets = [1, 10, 100]
        self.areaRng = [[0 ** 2, 1e5 ** 2], [0 ** 2, 128 ** 2], [ 128 ** 2, 256 ** 2], [256 ** 2, 1e5 ** 2]]
        self.areaRngLbl = ['all', 'small', 'medium', 'large']
        self.useCats = 1

    def setKpParams(self):
        self.vidIds = []
        self.catIds = []
        # np.arange causes trouble.  the data point on arange is slightly larger than the true value
        self.iouThrs = np.linspace(.5, 0.95, np.round((0.95 - .5) / .05) + 1, endpoint=True)
        self.recThrs = np.linspace(.0, 1.00, np.round((1.00 - .0) / .01) + 1, endpoint=True)
        self.maxDets = [20]
        self.areaRng = [[0 ** 2, 1e5 ** 2], [32 ** 2, 96 ** 2], [96 ** 2, 1e5 ** 2]]
        self.areaRngLbl = ['all', 'medium', 'large']
        self.useCats = 1

    def __init__(self, iouType='segm'):
        if iouType == 'segm' or iouType == 'bbox':
            self.setDetParams()
        elif iouType == 'keypoints':
            self.setKpParams()
        else:
            raise Exception('iouType not supported')
        self.iouType = iouType
        # useSegm is deprecated
        self.useSegm = None
