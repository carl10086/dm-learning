package com.ysz.dm.ddd.vshop.domain.core.common.media;

import java.io.Serializable;
import lombok.Getter;
import lombok.ToString;

/**
 * @author carl
 * @create 2022-09-09 2:39 PM
 **/
@ToString
@Getter
public final class Picture implements Serializable {

  private static final long serialVersionUID = -4333607368883814044L;

  private PictureId id;

  private String path;

  private int width;

  private int height;
}
