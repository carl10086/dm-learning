package com.ysz.dm.ddd.vshop.domain.domain.subscribe;

import com.ysz.dm.ddd.vshop.domain.domain.common.BaseEntity;
import com.ysz.dm.ddd.vshop.domain.domain.distribution.domain.AccountId;
import java.util.List;
import lombok.Getter;
import lombok.RequiredArgsConstructor;
import lombok.ToString;

/**
 * <pre>
 *
 *   account subscribe
 *
 * 1 个 account 可以在多个 pay channel 中同时存在订阅
 * </pre>
 *
 * @author carl.yu
 * @createAt 2022/11/9
 **/
@ToString
@Getter
@RequiredArgsConstructor
public class AccountSub extends BaseEntity<AccountId> {

  private final AccountId accountId;

  @Override
  protected AccountId id() {
    return accountId;
  }

  private List<AccountSubChannel> channels;

}
