package com.ysz.dm.ddd.vshop.domain.domain.subscribe;

import com.ysz.dm.ddd.vshop.domain.domain.common.BaseEntity;
import com.ysz.dm.ddd.vshop.domain.domain.distribution.domain.AccountId;
import java.util.List;
import lombok.Getter;
import lombok.RequiredArgsConstructor;
import lombok.Setter;
import lombok.ToString;

/**
 * <pre>
 * 1 个 account 在一个 用户触发的订阅周期 .
 *
 * - 开始于用户第一次签约
 * - 结束于用户取消签约
 * </pre>
 *
 * @author carl.yu
 * @createAt 2022/11/9
 **/
@Getter
@ToString
@RequiredArgsConstructor
public class SubChannelLifecycle extends BaseEntity<SubChannelLifecycleId> {

  private final SubChannelLifecycleId id;

  private final AccountId accountId;
  private final AccountSubChannelId subChannelId;


  /**
   * 一次签约
   */
  @Setter
  private SubSign sign;

  /**
   * 一次签约，多次续订
   */
  private transient List<SubRenewId> subRenews;

  private SubRenewLimitation latestLimitation;

  /**
   * 签约时间
   */
  private Long signAt;

  /**
   * 取消签约的时间
   */
  private Long cancelAt;


}
