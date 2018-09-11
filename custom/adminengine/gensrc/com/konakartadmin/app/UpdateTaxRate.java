package com.konakartadmin.app;

import com.konakartadmin.appif.*;
import com.konakartadmin.bl.KKAdmin;

/**
 *  The KonaKart Custom Engine - UpdateTaxRate - Generated by CreateKKAdminCustomEng
 */
@SuppressWarnings("all")
public class UpdateTaxRate
{
    KKAdmin kkAdminEng = null;

    /**
     * Constructor
     */
     public UpdateTaxRate(KKAdmin _kkAdminEng)
     {
         kkAdminEng = _kkAdminEng;
     }

     public int updateTaxRate(String sessionId, AdminTaxRate updateObj) throws KKAdminException
     {
         return kkAdminEng.updateTaxRate(sessionId, updateObj);
     }
}
