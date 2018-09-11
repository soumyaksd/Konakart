package com.konakartadmin.app;

import com.konakartadmin.bl.KKAdmin;

/**
 *  The KonaKart Custom Engine - DeleteAddressFormat - Generated by CreateKKAdminCustomEng
 */
@SuppressWarnings("all")
public class DeleteAddressFormat
{
    KKAdmin kkAdminEng = null;

    /**
     * Constructor
     */
     public DeleteAddressFormat(KKAdmin _kkAdminEng)
     {
         kkAdminEng = _kkAdminEng;
     }

     public int deleteAddressFormat(String sessionId, int id) throws KKAdminException
     {
         return kkAdminEng.deleteAddressFormat(sessionId, id);
     }
}
