package com.ysz.biz.http;

import lombok.Data;

/**
 * @author carl
 */
@Data
public class GelfHttpData {

  private String version = "1.1";

  private String _logger_name = "tstLogger";

  private String _source = "spark";

  private String _app = "sparkFeedRecommend";

  private Integer level = 1;

  private String _env = "release";

  private String full_message = "hello spark";

  private String short_message;

  private Integer timestamp = Integer.parseInt(System.currentTimeMillis() / 1000L + "");

}
