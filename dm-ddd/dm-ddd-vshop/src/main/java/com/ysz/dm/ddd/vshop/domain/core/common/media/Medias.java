package com.ysz.dm.ddd.vshop.domain.core.common.media;

import java.util.List;
import lombok.Getter;
import lombok.ToString;

/**
 * media 的聚合体. all 包含了 main .
 *
 * @author carl
 * @create 2022-09-15 11:44 AM
 **/
@ToString
@Getter
public class Medias {


  private Media main;

  private List<Media> all;

}
