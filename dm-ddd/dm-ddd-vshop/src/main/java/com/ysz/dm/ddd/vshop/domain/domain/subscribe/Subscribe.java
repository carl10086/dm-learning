package com.ysz.dm.ddd.vshop.domain.domain.subscribe;

import com.ysz.dm.ddd.vshop.domain.domain.common.BaseEntity;
import java.util.List;
import lombok.Getter;
import lombok.RequiredArgsConstructor;
import lombok.ToString;

/**
 * <pre>
 *  一个用户同时只能存在一个订阅 .  在所有的支付渠道上, 而且同时只能存在一个渠道 ..
 *
 *  TODO ? 如果用户反复的切换设备和订阅姿势.  -> . // 所以这个设计是有问题的 ...
 * </pre>
 *
 * @author carl
 * @create 2022-11-10 11:07 AM
 **/
@ToString
@Getter
@RequiredArgsConstructor
public class Subscribe extends BaseEntity<SubscribeId> {

  private final SubscribeId id;

  private List<SubscribeCycle> cycles;

}
