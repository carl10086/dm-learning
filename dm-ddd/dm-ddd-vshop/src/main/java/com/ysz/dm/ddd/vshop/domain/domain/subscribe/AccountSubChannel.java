package com.ysz.dm.ddd.vshop.domain.domain.subscribe;

import com.ysz.dm.ddd.vshop.domain.domain.common.BaseEntity;
import java.util.List;
import lombok.Getter;
import lombok.RequiredArgsConstructor;
import lombok.ToString;

/**
 * <pre>
 * account can only support one paychannel
 * </pre>
 *
 * @author carl.yu
 * @createAt 2022/11/9
 **/
@RequiredArgsConstructor
@ToString
@Getter
public class AccountSubChannel extends BaseEntity<AccountSubChannelId> {

  private final AccountSubChannelId id;


  private List<SubChannelLifecycleId> lifecycles;

}
