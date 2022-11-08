package com.ysz.dm.lib.lombok;

import lombok.Getter;
import lombok.NonNull;
import lombok.RequiredArgsConstructor;
import lombok.ToString;

/**
 * @author carl
 * @create 2022-10-25 7:16 PM
 **/
@ToString
@Getter
@RequiredArgsConstructor(staticName = "of")
public class SetterExample {

  private @NonNull Long userId;


}
