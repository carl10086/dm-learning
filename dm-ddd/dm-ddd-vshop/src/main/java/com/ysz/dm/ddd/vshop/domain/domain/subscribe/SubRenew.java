package com.ysz.dm.ddd.vshop.domain.domain.subscribe;

import com.ysz.dm.ddd.vshop.domain.domain.common.BaseEntity;
import lombok.Getter;
import lombok.RequiredArgsConstructor;
import lombok.ToString;

/**
 * <pre>
 * 代表用户在首单支付之后的 每次续订
 * </pre>
 *
 * @author carl.yu
 * @createAt 2022/11/9
 **/
@ToString
@Getter
@RequiredArgsConstructor
public class SubRenew extends BaseEntity<SubRenewId> {

  private final SubRenewId id;

  private final SubRenewLimitation limitation;

  private final SubRenewStatus status;


}
