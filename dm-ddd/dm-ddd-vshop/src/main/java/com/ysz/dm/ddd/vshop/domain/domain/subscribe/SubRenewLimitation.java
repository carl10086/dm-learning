package com.ysz.dm.ddd.vshop.domain.domain.subscribe;

import com.google.common.collect.Range;

/**
 * <pre>
 * 续订约束.  由支付渠道 + 上次的成功续订时间  + 签约时间决定 .
 *
 * 如果是支付宝, 签约时间是 7月10日, 上次的续订时间是 8月 15日. 这意味着 8 月 15日- 到9月15日的已经履约了.
 * - 所以下次的续订时间是. 9月? - ?
 *
 * 如果是 iap ? ...
 *
 *
 *
 * </pre>
 *
 * @author carl.yu
 * @createAt 2022/11/9
 **/
public record SubRenewLimitation(
    /*允许的续约时间*/
    Range<Long> renewAt
) {

}
