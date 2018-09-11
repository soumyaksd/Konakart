//
// (c) 2004-2015 DS Data Systems UK Ltd, All rights reserved.
//
// DS Data Systems and KonaKart and their respective logos, are 
// trademarks of DS Data Systems UK Ltd. All rights reserved.
//
// The information in this document is free software; you can redistribute 
// it and/or modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.
// 
// This software is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.
//

package com.konakartadmin.modules.ordertotal.subtotal;

import java.util.Date;

import com.konakart.util.Utils;
import com.konakartadmin.app.KKConfiguration;
import com.konakartadmin.bl.KKAdminBase;
import com.konakartadmin.modules.OrderTotalModule;

/**
 * SubTotal order total module
 * 
 */
public class SubTotal extends OrderTotalModule
{
    /**
     * @return the config key stub
     */
    public String getConfigKeyStub()
    {
        if (configKeyStub == null)
        {
            setConfigKeyStub(super.getConfigKeyStub() + "_SUBTOTAL");
        }
        return configKeyStub;       
    }


    public String getModuleTitle()
    {
        return getMsgs().getString("MODULE_ORDER_TOTAL_SUBTOTAL_TEXT_TITLE");
    }

    /**
     * @return the implementation filename - for compatibility with osCommerce we use the php name
     */
    public String getImplementationFileName()
    {
        return "ot_subtotal.php";
    }

    /**
     * @return the module code
     */
    public String getModuleCode()
    {
        return "ot_subtotal";
    }
    
    /**
     * @return an array of configuration values for this payment module
     */
    public KKConfiguration[] getConfigs()
    {
        if (configs == null)
        {
            configs = new KKConfiguration[2];
        }

        if (configs[0] != null && !Utils.isBlank(configs[0].getConfigurationKey()))
        {
            return configs;
        }

        Date now = KKAdminBase.getKonakartTimeStampDate();

        int i = 0;
        configs[i++] = new KKConfiguration("Display Sub-Total",
                "MODULE_ORDER_TOTAL_SUBTOTAL_STATUS", "true",
                "Do you want to display the order sub-total cost?", 6, 1, "",
                "choice('true'='true','false'='false')", now);
        configs[i++] = new KKConfiguration("Sort Order", "MODULE_ORDER_TOTAL_SUBTOTAL_SORT_ORDER",
                "10", "Sort order of display.", 6, 2, "", "", now);

        return configs;
    }
}
