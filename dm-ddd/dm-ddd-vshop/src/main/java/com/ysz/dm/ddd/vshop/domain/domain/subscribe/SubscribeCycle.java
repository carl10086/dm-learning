package com.ysz.dm.ddd.vshop.domain.domain.subscribe;

import com.ysz.dm.ddd.vshop.domain.domain.common.BaseEntity;
import java.util.List;
import lombok.Getter;
import lombok.RequiredArgsConstructor;
import lombok.ToString;

/**
 * <pre>
 * 一个用户的订阅周期
 *
 * 开始用户签约成功
 * 以前操作可以造成用户结束:
 *
 * - 用户手动取消 ;
 * - 商品升降级别, 导致需要重新签约 ;
 * - 如果是支付宝, 连续一段时间不续费导致的取消 ;
 *
 * </pre>
 *
 * 1. 只有有一次签约
 *
 * @author carl
 * @create 2022-11-10 11:12 AM
 **/
@ToString
@Getter
@RequiredArgsConstructor
public class SubscribeCycle extends BaseEntity<SubscribeCycleId> {

  private final SubscribeCycleId id;
  private final SubscribeSign sign;

  private final SubscribeId subscribeId;

  //TODO BE CONSIDER
  private final SubscribeSignId rootId;
  private final SubscribeCycleId parentId;

  private List<SubscribeRenew> renews;

  private SubscribeCycleUnit unit;

  private long cancelAt;

  public SubscribeRenewTimelimit calculate(long now) {
    return null;
  }

}
