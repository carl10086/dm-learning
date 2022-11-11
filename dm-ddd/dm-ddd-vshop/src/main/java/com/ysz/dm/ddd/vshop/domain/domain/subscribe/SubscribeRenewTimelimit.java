package com.ysz.dm.ddd.vshop.domain.domain.subscribe;

import com.google.common.collect.Range;
import lombok.Getter;
import lombok.RequiredArgsConstructor;
import lombok.ToString;

/**
 * 约束
 *
 * @author carl
 * @create 2022-11-10 11:41 AM
 **/
@ToString
@Getter
@RequiredArgsConstructor
public class SubscribeRenewTimelimit {

  private final Range<Long> limitation;

}
