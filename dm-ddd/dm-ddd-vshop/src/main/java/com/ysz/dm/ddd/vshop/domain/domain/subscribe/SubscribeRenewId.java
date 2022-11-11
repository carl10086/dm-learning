package com.ysz.dm.ddd.vshop.domain.domain.subscribe;

/**
 * <pre>
 *
 *   在某个 cycle 周期的唯一一个 时间点.
 *   例如是按月, 签约时间是1月5日 00:00:00
 *
 *   后续允许的 cycleNo 是 2月5日 00:00:00
 * </pre>
 *
 * @author carl
 * @create 2022-11-10 11:30 AM
 **/
public record SubscribeRenewId(
    long cycleNo,
    SubscribeCycleId id,
    SubscribeCycleUnit cycleUnit
) {

}
