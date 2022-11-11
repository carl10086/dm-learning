package com.ysz.dm.ddd.vshop.domain.domain.subscribe;

import com.ysz.dm.ddd.vshop.domain.domain.common.BaseEntity;
import lombok.Getter;
import lombok.ToString;

/**
 * <pre>
 *    月订阅 .
 *
 *    1. alipay 为例子, 签约时间是 1月5日, 首次订阅允许的时间是 1月5号
 *    - 下一次是 2月1日到 2月5日 .
 *
 *    2. 苹果为例子 .
 *
 *
 * </pre>
 *
 * @author carl
 * @create 2022-11-10 11:16 AM
 **/
@ToString
@Getter
public class SubscribeRenew extends BaseEntity<SubscribeRenewId> {

  private final SubscribeRenewId id;

  private final SubscribeRenewTimelimit limit;

  private long updateAt;

  public SubscribeRenew(
      long signAt,
      SubscribeCycleUnit unit,
      SubscribeCycleId id
  ) {
    this.id = null;
    this.limit = null;
  }


}
